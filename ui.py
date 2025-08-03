import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
load_dotenv()

from src.xml_parser import extract_roles_from_xml
from src.pdf_extractor_rag import RAGPDFExtractor
from src.role_comparer import RoleComparer
from config.config import (
    FUZZY_MATCH_THRESHOLD,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
)
from pinecone import Pinecone
import time

# ─── session state inits ──────────────────────────────────────────────────
if "validated" not in st.session_state:
    st.session_state.validated = False
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None
if "xml_bytes" not in st.session_state:
    st.session_state.xml_bytes = None
# persist extractor and chat history
if "pdf_extractor" not in st.session_state:
    st.session_state.pdf_extractor = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison(xml_file, pdf_file):
    """
    Extract XML roles, index the PDF in Pinecone, extract PDF roles,
    compare and show report. At the end we store the extractor in session_state.
    """
    # write files to disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp_xml:
        tmp_xml.write(xml_file.getvalue())
        xml_fp = tmp_xml.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(pdf_file.getvalue())
        pdf_fp = tmp_pdf.name

    try:
        # 1. XML Roles
        xml_roles = extract_roles_from_xml(xml_fp, '//role/text()')
        st.write(f"**Extracted XML Roles:** {xml_roles}")
        # 2. Init extractor
        extractor = RAGPDFExtractor()
        st.session_state.pdf_extractor = extractor
        # 3. Clear & re-index
        extractor.clear_pdf_data("uploaded-doc")
        extractor.process_pdf(pdf_fp, "uploaded-doc")
        # 4. Extract PDF roles via LLM
        pdf_roles = extractor.extract_roles_from_pdf(pdf_fp)
        st.write(f"**Extracted PDF Roles:** {pdf_roles}")
        # 5. Compare
        comparer = RoleComparer(fuzzy_threshold=FUZZY_MATCH_THRESHOLD)
        is_bad, matched, bad_roles = comparer.compare_roles(xml_roles, pdf_roles)
        # 6. Report
        st.write(f"- XML roles: {len(xml_roles)}  PDF roles: {len(pdf_roles)}")
        if bad_roles:
            st.error("Roles mismatch:")
            for r in bad_roles: st.write(f"- {r}")
        else:
            st.success("All PDF roles match XML!")
        st.session_state.validated = True

    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        os.remove(xml_fp)
        os.remove(pdf_fp)


st.set_page_config(page_title="VALIGENCE", layout="centered")
st.title("📄 VALIGENCE - A Role Validator Application")
st.markdown("Upload your XML (definitions) and PDF (to validate).")

uploaded_xml = st.file_uploader("XML file", type="xml")
uploaded_pdf = st.file_uploader("PDF file", type="pdf")

if uploaded_xml and uploaded_pdf:
    st.success("Files ready")
    if st.button("Start Validation"):
        # stash bytes for chat
        st.session_state.xml_bytes = uploaded_xml.getvalue()
        st.session_state.pdf_bytes = uploaded_pdf.getvalue()
        run_comparison(uploaded_xml, uploaded_pdf)
else:
    st.info("Please upload both an XML and a PDF.")

# ─── RAG CHAT UI ────────────────────────────────────────────────────────────
if st.session_state.validated:
    st.markdown("---")
    st.subheader("💬 Ask something about your PDF")

    # your extractor instance
    extractor = st.session_state.pdf_extractor

    # text input + button
    question = st.text_input("Your question", key="rag_q")
    if st.button("Ask PDF", key="rag_send"):
        if not question:
            st.warning("Type a question first.")
        else:
            with st.spinner("Fetching answer…"):
                # write out PDF bytes
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(st.session_state.pdf_bytes)
                    tmp_path = tmp.name

                # actually do the RAG query
                answer = extractor.query_pdf_for_roles_from_pinecone(
                    tmp_path, question
                )
                os.remove(tmp_path)

            # store and display
            st.session_state.chat_history.append((question, answer))

    # render full chat history
    for q, a in st.session_state.chat_history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")
        st.markdown("---")
