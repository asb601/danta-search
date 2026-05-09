# G-CHAT — API Endpoints

> **Base URL:** `https://genai.codeen.in.net/api`
> All requests require `Authorization: Bearer <token>` unless marked Public.

---

## Chat

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/chat/message` | User | Send a query, get full answer |
| POST | `/chat/message/stream` | User | Send a query, receive SSE token stream |
| GET | `/chat/conversations` | User | List all conversations |
| GET | `/chat/conversations/{id}` | User | Get a conversation with all messages |
| PATCH | `/chat/conversations/{id}` | User | Rename a conversation |
| DELETE | `/chat/conversations/{id}` | User | Archive / delete a conversation |

---

## Ingestion

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/files/upload-url` | Developer / Admin | Get a pre-signed Azure URL to upload a file |
| POST | `/files/confirm-upload` | Developer / Admin | Confirm the upload is complete |
| POST | `/chat/ingest` | Developer / Admin | Trigger AI ingestion on confirmed files |
| POST | `/admin/reingest-all` | Admin | Re-ingest all failed or stale files |
| POST | `/admin/retry-parquet` | Admin | Retry Parquet conversion for missing blobs |
| GET | `/admin/missing-parquet` | Admin | List files with missing Parquet blobs |

---

## Files CRUD

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/folders/{id}/contents` | User | List files and subfolders (use `root` for top level) |
| PATCH | `/files/{id}/rename` | Developer / Admin | Rename a file |
| PATCH | `/files/{id}/move` | Developer / Admin | Move a file to another folder |
| DELETE | `/files/{id}` | Developer / Admin | Delete a file |

---

## Folders CRUD

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/folders` | Developer / Admin | Create a folder |
| PATCH | `/folders/{id}` | Developer / Admin | Rename or tag a folder |
| DELETE | `/folders/{id}` | Developer / Admin | Delete a folder and its files |
| PATCH | `/admin/folders/{id}/domain` | Admin | Tag a folder with a domain label |

---

## Containers CRUD

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/containers` | Developer / Admin | List containers |
| POST | `/containers` | Developer / Admin | Create a container |
| POST | `/containers/{id}/sync` | Developer / Admin | Sync files from Azure Blob Storage |
| DELETE | `/containers/{id}` | Developer / Admin | Delete a container |

---

## Users & Access

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/auth/google/login` | Public | Start Google OAuth login |
| GET | `/auth/google/callback` | Public | OAuth callback — issues JWT |
| GET | `/auth/me` | User | Get current user info |
| GET | `/users` | Admin | List all users |
| PATCH | `/users/{id}/role` | Admin | Set user role (admin / developer / user) |
| DELETE | `/users/{id}` | Admin | Delete a user |
| GET | `/users/domains` | User | List all available domain tags |
| PATCH | `/users/me/domains` | User | Set your own domain filter |
| PATCH | `/admin/users/{id}/domains` | Admin | Set domain restrictions for any user |
| POST | `/access-requests/me` | User | Submit an access request |
| GET | `/access-requests` | Admin | List all access requests |
| PATCH | `/access-requests/{id}/approve` | Admin | Approve an access request |
| PATCH | `/access-requests/{id}/decline` | Admin | Decline an access request |

---

## Admin

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/admin/cost-summary` | Admin | LLM token usage and estimated cost |
