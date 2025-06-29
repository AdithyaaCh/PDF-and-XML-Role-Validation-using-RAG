# AI Role Validator: XML to PDF Job Role Comparison

<img width="820" alt="image" src="https://github.com/user-attachments/assets/b4960c68-8d89-432d-89e3-a106cb630c84" />



> A smart tool to compare structured job role definitions (from XML) with roles found in unstructured PDF job descriptions using AI, RAG, and fuzzy logic.

---

## Overview

The **AI Role Validator** is an intelligent Python application designed to automate and streamline the validation of job roles.

It compares a definitive list of roles specified in a structured **XML file** against roles extracted from unstructured **PDF documents**. Using **Generative AI (Google Gemini)**, **Retrieval-Augmented Generation (RAG)**, and **fuzzy string matching**, it ensures consistency, highlights discrepancies, and eliminates hours of manual effort.

---

## Features

- XML Role Extraction – Parses structured XML files for the authoritative list of job roles.
- PDF Content Extraction – Reads text and tables from PDFs using `PyMuPDF`.
- LLM-Powered Role Extraction – Uses **Google Gemini AI** to extract roles from messy, unstructured text.
- RAG-Based Enhancement – Integrates **Pinecone** vector DB for chunk-level retrieval from PDFs.
- Fuzzy Matching – Accounts for typos, abbreviations, and formatting inconsistencies.
- Validation Report – Clearly classifies:
  - Matched Roles (exact & fuzzy)
  - Unmatched/Incorrect Roles
- Configurable Thresholds – Tune fuzzy matching sensitivity.

---

## How Fuzzy Matching Works

Used to catch **typos** and **minor word-level errors**.

### 1️⃣ Levenshtein Distance – `fuzz.ratio()`

> **Example:**
> XML: `Tester`
> PDF: `Tester`
> → Edit distance = 1 substitution
> → Similarity ≈ **83.33%**

### 2️⃣ Ratcliff-Obershelp – `fuzz.partial_ratio()`

Used to catch **abbreviations** or **substring matches**.

> **Example:**
> XML: `Software Engineer`
> PDF: `Software Eng.`
> → Partial ratio = **100%**
> (as it's a near-perfect subset)

The intelligent combination of both methods ensures robust matching, even across formatting variations and abbreviations.

## 🛠️ Technologies Used

- **Python 3.9+**
- **Google Gemini API**
- **Pinecone Vector DB**
- **PyMuPDF (fitz)**
- **thefuzz (fuzzywuzzy)**
- **lxml**
- **langchain-text-splitters**
- **python-dotenv**

