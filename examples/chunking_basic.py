"""Chunk a short US statute excerpt with the legal-aware chunker.

Shows: RecursiveLegalChunker splitting text on section boundaries, with token counts,
detected document type, section labels and neighbor links on every chunk.
Needs: nothing. The `chunking` extra (`pip install "legal-rag[chunking]"`) adds tiktoken for
exact token counts; without it a whitespace approximation is used. No API keys, no network.

Run: python examples/chunking_basic.py
"""

import asyncio

from legal_rag.chunking import RecursiveLegalChunker

STATUTE = """\
CHAPTER 12. CONSUMER DATA PROTECTION

Section 12-101. Definitions.
In this chapter: (1) "Personal data" means information that identifies, relates to, or is
reasonably linkable to an individual consumer. (2) "Controller" means a person that, alone or
jointly with others, determines the purposes and means of processing personal data.
(3) "Processor" means a person that processes personal data on behalf of a controller.

Section 12-102. Consumer rights.
A consumer may, by written request to a controller, (1) confirm whether the controller is
processing the consumer's personal data, (2) obtain a copy of that data, and (3) require the
controller to delete personal data provided by or obtained about the consumer.

Section 12-103. Controller duties.
A controller shall limit the collection of personal data to what is adequate, relevant, and
reasonably necessary for the disclosed purpose. A controller shall not process sensitive data
without the consumer's consent. A controller shall respond to a verified request within
forty-five days of receipt, and may extend that period once by forty-five days when reasonably
necessary, taking into account the complexity and number of the requests.

Section 12-104. Enforcement.
The Attorney General has exclusive authority to enforce this chapter. Before initiating an
action, the Attorney General shall give the controller thirty days to cure any violation.
"""


async def main() -> None:
    chunker = RecursiveLegalChunker(chunk_size=120, chunk_overlap=20, min_chunk_size=20)
    result = await chunker.chunk(
        STATUTE, document_id="demo-chapter-12", document_metadata={"title": "Consumer Data Act"}
    )
    print(f"{result.total_chunks} chunks, {result.total_tokens} tokens in total")
    print("(each chunk starts with a contextual header, then the text)\n")
    for chunk in result.chunks:
        doc_type = getattr(chunk, "document_type", None)
        section = getattr(chunk, "section_type_universal", None)
        print(
            f"[{chunk.chunk_index}] {chunk.token_count:>3} tokens"
            f" | type={getattr(doc_type, 'value', doc_type)}"
            f" | section={getattr(section, 'value', section)}"
            f" | chars {chunk.start_char}-{chunk.end_char}"
        )
        print("    " + " ".join(chunk.content.split())[:90] + "...")
    linked = sum(1 for c in result.chunks if c.next_chunk_id)
    print(f"\n{linked} chunks link to a next chunk (prev/next ids are set).")


if __name__ == "__main__":
    asyncio.run(main())
