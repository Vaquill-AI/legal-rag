# ruff: noqa: BLE001, S110, RUF012
# Ingestion degrades per document instead of failing a batch, hence the broad excepts.
"""
Image extractor: OCR for PNG, JPEG, TIFF, BMP, GIF, WebP and HEIC.

Uses two injected services, both optional:
- ``VisionClient``: primary path. One call transcribes any text and describes the scene.
- ``OcrClient``: fallback when the vision client is missing or returns nothing.

GIF, BMP, WEBP and similar inputs are transcoded to PNG (and oversized images downscaled)
before either call, because most OCR and vision endpoints accept only JPEG, PNG and a few
more formats.
"""

import asyncio
import io

from legal_rag.ingestion._log import get_logger
from legal_rag.ingestion.extractors.base import BaseExtractor, ExtractedContent
from legal_rag.ingestion.protocols import OcrClient, VisionClient

logger = get_logger("legal_rag.ingestion.extractors.image")

NO_TEXT_SENTINEL = "NO_TEXT_FOUND"

NO_CLIENT_ERROR = (
    "Could not read this image: no OCR or vision client is configured. "
    "Pass an OcrClient or a VisionClient to ImageExtractor (or MultiFormatIngestor) "
    "to extract text from images, or upload a text-based version of the document."
)

DEFAULT_VISION_PROMPT = (
    "Do TWO things:\n"
    "1. Transcribe ALL readable text in the image exactly as it appears (documents, labels, "
    "signs, captions), preserving formatting, line breaks and structure.\n"
    "2. If the image is a photo, map, diagram or scene rather than a plain text document, ALSO "
    "describe its visual content factually and specifically under a heading 'Visual "
    "description'. Describe only what is visible; do not speculate about causes or meaning.\n"
    f"Respond with exactly {NO_TEXT_SENTINEL} only if the image is genuinely blank or "
    "unintelligible."
)

PIL_AVAILABLE = False
try:
    from PIL import Image, ImageOps

    PIL_AVAILABLE = True
except ImportError:
    logger.info("pillow_not_installed", message="Image transcoding disabled")

# iPhone photos default to HEIC/HEIF, which Pillow cannot open on its own.
# Registering the pillow-heif opener lets Image.open() read them so they can be
# transcoded to PNG for the vision model. Optional: absence just means HEIC
# uploads fail to open (unchanged from before).
HEIF_AVAILABLE = False
if PIL_AVAILABLE:
    try:
        # Imported from the submodule, not the package top level: a newer
        # pillow-heif stopped re-exporting this name from `__init__`, which
        # pyright treats as a private import even though the runtime symbol
        # is unchanged.
        from pillow_heif.as_plugin import register_heif_opener

        register_heif_opener()
        HEIF_AVAILABLE = True
    except ImportError:
        logger.info(
            "pillow_heif_not_installed",
            message="HEIC/HEIF images cannot be decoded; install pillow-heif",
        )


class ImageExtractor(BaseExtractor):
    """
    Extract text from images through an injected vision client and OCR client.

    Supported formats:
    - PNG, JPEG/JPG, TIFF/TIF (passed through)
    - BMP, GIF, WebP, HEIC (transcoded to PNG first)

    Requirements:
    - At least one of ``vision_client`` or ``ocr_client`` for any text to be returned
    - Pillow installed for size detection and transcoding (optional)

    Args:
        ocr_client: ``OcrClient`` used when vision is unavailable or returns nothing.
        vision_client: ``VisionClient`` tried first.
        vision_prompt: Instruction sent with the image.
    """

    def __init__(
        self,
        ocr_client: OcrClient | None = None,
        vision_client: VisionClient | None = None,
        *,
        vision_prompt: str = DEFAULT_VISION_PROMPT,
    ) -> None:
        self.ocr_client = ocr_client
        self.vision_client = vision_client
        self.vision_prompt = vision_prompt

    SUPPORTED = {
        "png",
        "jpeg",
        "jpg",
        "tiff",
        "tif",
        "bmp",
        "gif",
        "webp",
        "heic",  # iPhone default; decoded via pillow-heif, then transcoded to PNG
        "heif",
    }

    # Per-format extensions Mistral OCR understands directly.
    _MISTRAL_PASSTHROUGH = {"png", "jpg", "jpeg", "tiff", "tif", "avif"}

    # Formats accepted by BOTH the vision API (PNG/JPEG/GIF/WEBP) and Mistral
    # OCR (PNG/JPEG/TIFF) with no transcode -- i.e. the safe intersection. The
    # one normalized output feeds both the vision primary and the OCR fallback,
    # so anything outside this set (GIF, WEBP, TIFF, BMP) is transcoded to PNG.
    _NATIVE_PASSTHROUGH = {"png", "jpg", "jpeg"}

    # The vision API caps images at 20 MB and downsizes anything larger to a
    # 768px short side, so a huge upload just wastes tokens (and can exceed the
    # limit). Cap the long side so we stay well under 20 MB with headroom for
    # the OCR fallback's fidelity.
    _MAX_VISION_DIM = 2048

    async def extract(
        self,
        file_content: bytes,
        filename: str,
        **kwargs,
    ) -> ExtractedContent:
        """Extract content from a single image, vision-model first.

        An image is a single piece (not a multi-page document), so the vision
        model is the primary extractor: it transcribes any text AND describes the
        visual scene (road/lane layout, signs, vehicle/property damage, diagrams)
        in one call. Mistral OCR is the fallback when the vision model is
        unavailable (no vision client) or returns nothing.
        """
        from legal_rag.ingestion.text_cleaner import clean_ocr_artifacts

        ext = self.get_extension(filename)

        ocr_bytes = file_content
        ocr_filename = filename
        width: int | None = None
        height: int | None = None
        mode: str | None = None

        # Normalize for the vision model (primary path): OpenAI vision accepts
        # only PNG/JPEG/GIF/WEBP and caps images at 20 MB, downsizing anything
        # larger to a 768px short side anyway. So we transcode non-native formats
        # (TIFF/BMP/...) to PNG and downscale oversized images. The normalized
        # bytes work for the Mistral OCR fallback too.
        if PIL_AVAILABLE:
            try:
                img = Image.open(io.BytesIO(file_content))
                # Phone photos carry EXIF orientation; without baking it in, the
                # model sees the image sideways. Only re-save when an actual
                # rotation is present (tag 274 / Orientation, values 2-8).
                needs_exif_bake = False
                try:
                    orientation = img.getexif().get(0x0112)
                    if orientation and orientation != 1:
                        img = ImageOps.exif_transpose(img)
                        needs_exif_bake = True
                except Exception:
                    pass
                width, height = img.size
                mode = img.mode
                needs_transcode = ext not in self._NATIVE_PASSTHROUGH or needs_exif_bake
                needs_resize = max(width, height) > self._MAX_VISION_DIM
                if needs_transcode or needs_resize:
                    work = img.copy()
                    if needs_resize:
                        work.thumbnail((self._MAX_VISION_DIM, self._MAX_VISION_DIM))
                    buf = io.BytesIO()
                    work.convert("RGB").save(buf, format="PNG")
                    ocr_bytes = buf.getvalue()
                    ocr_filename = f"{filename.rsplit('.', 1)[0]}.png"
                    logger.debug(
                        "image_normalized_for_vision",
                        original_filename=filename,
                        normalized_filename=ocr_filename,
                        original_ext=ext,
                        transcoded=needs_transcode,
                        resized=needs_resize,
                        new_size=work.size,
                    )
            except Exception as e:
                logger.warning(
                    "image_open_failed_before_extract",
                    filename=filename,
                    error=str(e),
                )

        # Images are a SINGLE piece (not a multi-page document), so the vision
        # model is the PRIMARY extractor: one call both transcribes any text AND
        # describes the visual scene (road/lane layout, signs, vehicle/property
        # damage, diagrams) that OCR alone cannot convey. We pass the normalized
        # (PNG-transcoded) bytes so uncommon formats reach the vision API too.
        # Mistral OCR is the fallback for when vision is unavailable or returns
        # nothing (no OpenAI key, API failure, or a scan the model declines).
        vision_result = await self._vision_llm_fallback(ocr_bytes, ocr_filename)
        if vision_result:
            cleaned_vision = clean_ocr_artifacts(vision_result, preserve_structure=True)
            logger.info(
                "image_vision_primary_success",
                filename=filename,
                chars=len(vision_result),
            )
            v_metadata: dict = {"extraction_method": "vision_llm"}
            if width is not None:
                v_metadata["width"] = width
            if height is not None:
                v_metadata["height"] = height
            if mode is not None:
                v_metadata["mode"] = mode
            return ExtractedContent(
                text=vision_result,
                text_clean=cleaned_vision,
                page_count=1,
                metadata=v_metadata,
                pages=[{"page_number": 1, "text": cleaned_vision}],
                success=True,
            )

        # Fallback: Mistral OCR (vision unavailable or produced nothing).
        ocr_text: str = ""
        if self.ocr_client is not None:
            try:
                result = await asyncio.to_thread(
                    self.ocr_client.ocr,
                    ocr_bytes,
                    ocr_filename,
                    include_images=False,
                )
                ocr_text = ((result or {}).get("text") or "").strip()
            except Exception as e:
                logger.warning(
                    "ocr_failed_for_image",
                    filename=filename,
                    error=str(e),
                )

        if ocr_text:
            cleaned = clean_ocr_artifacts(ocr_text, preserve_structure=True)
            logger.debug(
                "image_ocr_fallback_used",
                filename=filename,
                chars=len(ocr_text),
                vendor="ocr_client",
            )
            o_metadata: dict = {"extraction_method": "ocr"}
            if width is not None:
                o_metadata["width"] = width
            if height is not None:
                o_metadata["height"] = height
            if mode is not None:
                o_metadata["mode"] = mode
            return ExtractedContent(
                text=ocr_text,
                text_clean=cleaned,
                page_count=1,
                metadata=o_metadata,
                pages=[{"page_number": 1, "text": cleaned}],
                success=True,
            )

        # An image that no client could read must not look like a successful empty read.
        if self.vision_client is None and self.ocr_client is None:
            return ExtractedContent(
                text="",
                success=False,
                error=NO_CLIENT_ERROR,
                metadata={"width": width, "height": height, "mode": mode},
            )

        # Neither pipeline produced anything (a genuinely blank / unreadable image).
        return ExtractedContent(
            text="",
            success=True,
            metadata={
                "width": width,
                "height": height,
                "mode": mode,
                "note": "No text detected by the vision model or OCR",
            },
        )

    async def _vision_llm_fallback(self, file_content: bytes, filename: str) -> str | None:
        """
        Primary image extractor: ask the vision client to transcribe text and describe the
        scene.

        Returns:
            Extracted text and description, or None if no vision client is configured, the
            call failed, or the image is blank.
        """
        if self.vision_client is None:
            logger.debug("vision_llm_skip_no_client", filename=filename)
            return None
        try:
            ext = self.get_extension(filename)
            mime_map = {
                "png": "image/png",
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "gif": "image/gif",
                "webp": "image/webp",
                "bmp": "image/bmp",
                "tiff": "image/tiff",
                "tif": "image/tiff",
            }
            mime_type = mime_map.get(ext, "image/png")
            raw = await self.vision_client.describe_image(
                file_content, mime_type, prompt=self.vision_prompt
            )
            result_text = (raw or "").strip()
            if not result_text or result_text == NO_TEXT_SENTINEL:
                logger.debug("vision_llm_no_text", filename=filename)
                return None
            logger.info("vision_llm_extracted_text", filename=filename, chars=len(result_text))
            return result_text
        except Exception as e:
            logger.warning("vision_llm_fallback_failed", filename=filename, error=str(e))
            return None
