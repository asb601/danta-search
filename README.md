# danta-search

danta-search is an enterprise semantic search and data intelligence application for teams that need to ask business questions across uploaded operational files without giving the system direct access to production databases.

The application lets users upload files, process them into query-ready datasets, discover relationships between tables, and ask natural-language questions over their data. Behind the scenes, it combines metadata extraction, semantic role detection, vector search, graph relationships, SQL execution, and planner-safe context assembly so answers are grounded in actual files, columns, and approved relationships.

## What It Does

- Lets teams ingest CSV, Excel, and Parquet-style business data into a controlled workspace.
- Converts uploaded data into efficient query formats for faster analysis.
- Builds searchable metadata from file names, columns, descriptions, semantic roles, and embeddings.
- Finds likely relationships between files, such as invoice to vendor, payment to invoice, or shipment to receipt.
- Uses graph-aware retrieval to bring the right workflow context into the planner before SQL reasoning begins.
- Provides a chat interface where users can ask business questions instead of manually joining tables.
- Keeps source data isolated from the client's live operational systems.

## Why It Matters

Most enterprise data questions are not single-table questions. A user asking about invoice mismatches may need invoice lines, purchase orders, vendor master data, goods receipts, and payments. A user asking about delayed delivery may need shipment status, carrier data, expected delivery dates, and receipt records.

danta-search is designed for that kind of workflow-level reasoning. It does not only retrieve files by keyword. It assembles the surrounding semantic business context that the planner needs before it starts answering.

## Key Capabilities

### Semantic Retrieval

danta-search uses multiple retrieval signals, including keyword search, fuzzy matching, embeddings, graph expansion, and reciprocal-rank fusion. This helps the system find useful files even when users do not know exact table or column names.

### Workflow-Aware Context Assembly

The system identifies semantic domains such as invoices, vendors, payments, purchase orders, shipments, carriers, and receipts from column roles and file relationships. It can expand the planner context when a workflow is incomplete.

### Planner-Safe SQL Reasoning

The planner receives validated file context, approved join paths, available Parquet paths, semantic role hints, and workflow topology notes. This reduces hallucinated joins and keeps analysis tied to actual data.

### Enterprise Data Isolation

danta-search works on uploaded or connected file storage data. It does not require access to a client's production application database. The platform stores its own metadata, embeddings, permissions, and relationship graph separately.

### Admin and User Experience

The client application includes onboarding, file management, chat, profile, and admin surfaces. The goal is to make data exploration usable for business users while preserving enough operational control for technical teams.

## Current Status

The application already supports file ingestion, metadata enrichment, vector retrieval, semantic relationship detection, workflow-aware shortlist expansion, and natural-language analysis over approved data context.

The most recent stabilization work focused on semantic workflow assembly. The system now avoids falsely reporting incomplete workflow context as complete, and it can infer adjacent workflow domains through semantic roles, approved graph topology, and retrieval evidence.

## Future Improvements

The next major planned improvement is a dedicated PDF agent. This is still in progress. The intent is to let danta-search handle document-heavy workflows such as contracts, invoices, reports, policies, statements, and other PDF-based business records with the same level of semantic grounding used for structured files.

Planned PDF-agent capabilities include:

- PDF ingestion and page-aware parsing.
- Table and form extraction.
- Document chunk grounding with citations.
- Cross-document semantic search.
- Linking PDF evidence to structured data when both describe the same workflow.
- Planner-safe answers that can reference both structured tables and PDF source material.

## Project Direction

danta-search is being built as a semantic workflow orchestration engine, not a simple retrieval chatbot. The long-term goal is to give users complete business context before reasoning begins, so answers reflect the workflow behind the question rather than only the closest matching file.
