"""Extract text from in-memory legal documents with the DocumentRouter.

Shows: routing a plain-text opinion, an HTML opinion and (when python-docx is installed)
a generated DOCX memo to a parser, then reading the unified IngestorResult.
Needs: nothing for the .txt file. The .html and .docx files need the `ingestion-office`
extra (`pip install "legal-rag[ingestion-office]"`); they are skipped with a note otherwise.
No API keys, no network, no files on disk.

Run: python examples/ingestion_basic.py
"""

import asyncio
import io

from legal_rag.ingestion import DocumentRouter

OPINION = (
    "SUPREME COURT OF THE UNITED STATES\n\n"
    "Marbury v. Madison, 5 U.S. (1 Cranch) 137 (1803)\n\n"
    "MARSHALL, C.J., delivered the opinion of the Court.\n\n"
    "It is emphatically the province and duty of the judicial department to say what the law "
    "is. Those who apply the rule to particular cases must of necessity expound and interpret "
    "that rule. If two laws conflict with each other, the courts must decide on the operation "
    "of each.\n"
)

HTML = (
    "<html><head><title>Marbury v. Madison</title></head><body>"
    "<h1>Marbury v. Madison</h1><p>5 U.S. 137 (1803)</p>"
    "<p>A law repugnant to the Constitution is void, and courts, as well as other "
    "departments, are bound by that instrument.</p></body></html>"
)


def make_docx() -> bytes | None:
    try:
        from docx import Document
    except ImportError:
        return None
    doc = Document()
    doc.add_heading("Memorandum: Judicial Review", level=1)
    doc.add_paragraph(
        "Under Marbury, federal courts may strike down statutes that conflict with the Constitution."
    )
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


async def main() -> None:
    router = DocumentRouter()
    files = {"opinion.txt": OPINION.encode(), "opinion.html": HTML.encode()}
    docx = make_docx()
    if docx is not None:
        files["memo.docx"] = docx
    else:
        print(
            "python-docx not installed, skipping memo.docx (install legal-rag[ingestion-office])\n"
        )

    for filename, data in files.items():
        routed = await router.route_and_parse(
            file_content=data, filename=filename, document_id=filename
        )
        doc = routed.ingestor_result
        print(f"{filename}: parser={routed.parser_used.value} success={doc.success}")
        if not doc.success:
            print(f"  error: {doc.error}\n")
            continue
        print(f"  type={doc.document_type.value} pages={doc.page_count} chars={len(doc.content)}")
        if routed.routing_decision:
            print(f"  routing: {routed.routing_decision.reasoning}")
        print("  text: " + " ".join(doc.content.split())[:100] + "...\n")


if __name__ == "__main__":
    asyncio.run(main())
