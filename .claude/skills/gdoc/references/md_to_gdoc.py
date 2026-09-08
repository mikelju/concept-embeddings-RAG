#!/usr/bin/env python3
"""Convert a Markdown file to a Google Doc, preserving rendered formatting.

Portable script — works from any project that has credentials.json in its root.

Usage:
    python .claude/skills/gdoc/references/md_to_gdoc.py archivo.md
    python .claude/skills/gdoc/references/md_to_gdoc.py archivo.md --title "Mi Documento"
    python .claude/skills/gdoc/references/md_to_gdoc.py archivo.md --folder-id 1abc...xyz
    python .claude/skills/gdoc/references/md_to_gdoc.py archivo.md --share otro@email.com
    python .claude/skills/gdoc/references/md_to_gdoc.py archivo.md --credentials /path/to/creds.json

Dependencies: python-docx, markdown-it-py, google-api-python-client, google-auth
"""

import argparse
import os
import sys
from pathlib import Path

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

# Import the MD→DOCX converter from the same directory
sys.path.insert(0, str(Path(__file__).parent))
from md_to_docx import convert_md_to_docx

# Configure with the GDOC_SHARE_EMAIL environment variable. Deliberately not a
# hardcoded address: this repository is public.
DEFAULT_SHARE_EMAIL = os.environ.get("GDOC_SHARE_EMAIL", "")
SCOPES = ["https://www.googleapis.com/auth/drive"]


def find_credentials() -> Path:
    """Find credentials.json by walking up from CWD to find the project root."""
    # 1. Check CWD
    cwd = Path.cwd()
    candidate = cwd / "credentials.json"
    if candidate.exists():
        return candidate

    # 2. Walk up looking for credentials.json
    for parent in cwd.parents:
        candidate = parent / "credentials.json"
        if candidate.exists():
            return candidate

    # 3. Check env var
    env_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    return cwd / "credentials.json"  # will fail with a clear message


def get_drive_service(credentials_path: Path):
    """Authenticate with Google Drive using Service Account credentials."""
    if not credentials_path.exists():
        print(
            f"ERROR: No se encuentra {credentials_path}\n"
            "Setup:\n"
            "  1. Coloca credentials.json (Service Account) en la raiz del proyecto\n"
            "  2. Asegurate de que Google Drive API esta habilitada en tu proyecto GCP\n"
            "  3. Alternativa: usa --credentials /ruta/a/credentials.json",
            file=sys.stderr,
        )
        sys.exit(1)

    creds = Credentials.from_service_account_file(str(credentials_path), scopes=SCOPES)
    return build("drive", "v3", credentials=creds)


def upload_to_gdoc(service, docx_path: Path, title: str, folder_id: str | None = None) -> str:
    """Upload a .docx file to Google Drive, converting it to a Google Doc.

    Returns the file ID of the created Google Doc.
    """
    file_metadata = {
        "name": title,
        "mimeType": "application/vnd.google-apps.document",
    }
    if folder_id:
        file_metadata["parents"] = [folder_id]

    media = MediaFileUpload(
        str(docx_path),
        mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        resumable=True,
    )

    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id",
    ).execute()

    return file["id"]


def share_document(service, file_id: str, email: str):
    """Share a Google Drive file with the given email as editor."""
    service.permissions().create(
        fileId=file_id,
        body={
            "type": "user",
            "role": "writer",
            "emailAddress": email,
        },
        sendNotificationEmail=False,
    ).execute()


def main():
    parser = argparse.ArgumentParser(
        description="Convert a Markdown file to a Google Doc"
    )
    parser.add_argument("file", help="Markdown file to convert")
    parser.add_argument(
        "--title",
        default=None,
        help="Google Doc title (default: filename without extension)",
    )
    parser.add_argument(
        "--folder-id",
        default=None,
        help="Google Drive folder ID to upload into",
    )
    parser.add_argument(
        "--share",
        default=DEFAULT_SHARE_EMAIL,
        help=f"Email to share with (default: {DEFAULT_SHARE_EMAIL})",
    )
    parser.add_argument(
        "--no-share",
        action="store_true",
        help="Don't share the document automatically",
    )
    parser.add_argument(
        "--keep-docx",
        action="store_true",
        help="Keep the intermediate .docx file",
    )
    parser.add_argument(
        "--credentials",
        default=None,
        help="Path to Google Service Account credentials.json",
    )
    args = parser.parse_args()

    # Validate input
    md_path = Path(args.file)
    if not md_path.exists():
        print(f"ERROR: Archivo no encontrado: {md_path}", file=sys.stderr)
        return 1

    title = args.title or md_path.stem.replace("_", " ").replace("-", " ").title()

    # Resolve credentials
    credentials_path = Path(args.credentials) if args.credentials else find_credentials()

    # Step 1: MD -> DOCX (use .tmp/ in CWD)
    tmp_dir = Path.cwd() / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Convirtiendo {md_path.name} a DOCX...")
    docx_path = convert_md_to_docx(md_path, tmp_dir)
    print(f"  OK  DOCX generado: {docx_path}")

    # Step 2: Upload to Google Drive
    try:
        print("  Subiendo a Google Drive...")
        service = get_drive_service(credentials_path)
        file_id = upload_to_gdoc(service, docx_path, title, args.folder_id)
        doc_url = f"https://docs.google.com/document/d/{file_id}/edit"
        print(f"  OK  Google Doc creado: {doc_url}")
    except HttpError as e:
        if "Google Drive API has not been used" in str(e) or e.resp.status == 403:
            print(
                "ERROR: Google Drive API no esta habilitada.\n"
                "Habilitala en: https://console.cloud.google.com/apis/library/drive.googleapis.com",
                file=sys.stderr,
            )
        else:
            print(f"ERROR Google API: {e}", file=sys.stderr)
        return 1

    # Step 3: Share
    if not args.no_share:
        try:
            share_document(service, file_id, args.share)
            print(f"  OK  Compartido con {args.share}")
        except HttpError as e:
            print(
                f"  WARN  No se pudo compartir con {args.share}: {e}\n"
                f"  El documento sigue accesible en: {doc_url}",
                file=sys.stderr,
            )

    # Step 4: Cleanup
    if not args.keep_docx:
        docx_path.unlink(missing_ok=True)

    print(f"\n  URL: {doc_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
