"""Small Google Drive API helper for a shared folder and a service account.

Credentials should be supplied at runtime (for example through Colab Secrets),
never committed to the repository. The helper mirrors folders by relative path.
"""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path


FOLDER_MIME = "application/vnd.google-apps.folder"


def build_service(credentials_json: str):
    """Build a Drive v3 client from the JSON text of a service-account key."""
    try:
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(
            "Install google-auth and google-api-python-client to use Drive API."
        ) from exc

    info = json.loads(credentials_json)
    credentials = Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def _list_children(service, folder_id: str):
    page_token = None
    while True:
        response = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="nextPageToken,files(id,name,mimeType,size,md5Checksum)",
            pageSize=1000,
            pageToken=page_token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        yield from response.get("files", [])
        page_token = response.get("nextPageToken")
        if not page_token:
            break


def download_folder(service, folder_id: str, destination: str | Path) -> list[Path]:
    """Download a shared Drive folder recursively, preserving its tree."""
    from googleapiclient.http import MediaIoBaseDownload

    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    def visit(remote_id: str, local_dir: Path):
        for item in _list_children(service, remote_id):
            local_path = local_dir / item["name"]
            if item["mimeType"] == FOLDER_MIME:
                local_path.mkdir(parents=True, exist_ok=True)
                visit(item["id"], local_path)
                continue
            request = service.files().get_media(fileId=item["id"], supportsAllDrives=True)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with local_path.open("wb") as stream:
                downloader = MediaIoBaseDownload(stream, request, chunksize=8 * 1024 * 1024)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
            downloaded.append(local_path)

    visit(folder_id, root)
    return downloaded


def ensure_folder(service, name: str, parent_id: str) -> str:
    """Find or create a child folder and return its Drive ID."""
    escaped = name.replace("'", "\\'")
    response = service.files().list(
        q=f"'{parent_id}' in parents and name = '{escaped}' and "
          f"mimeType = '{FOLDER_MIME}' and trashed = false",
        fields="files(id,name)",
        pageSize=100,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    matches = response.get("files", [])
    if matches:
        return matches[0]["id"]
    created = service.files().create(
        body={"name": name, "mimeType": FOLDER_MIME, "parents": [parent_id]},
        fields="id",
        supportsAllDrives=True,
    ).execute()
    return created["id"]


def find_folder(service, root_folder_id: str, name: str) -> str:
    """Find a uniquely named folder below a shared root; fail on ambiguity."""
    matches: list[str] = []
    pending = [root_folder_id]
    while pending:
        parent_id = pending.pop()
        for item in _list_children(service, parent_id):
            if item["mimeType"] != FOLDER_MIME:
                continue
            if item["name"] == name:
                matches.append(item["id"])
            pending.append(item["id"])
    if not matches:
        raise FileNotFoundError(f"Folder '{name}' tidak ditemukan di Drive folder {root_folder_id}.")
    if len(matches) > 1:
        raise ValueError(f"Folder '{name}' tidak unik di Drive folder {root_folder_id}; gunakan ID eksplisit.")
    return matches[0]


def upload_folder(service, source: str | Path, parent_folder_id: str) -> list[str]:
    """Upload a local directory recursively beneath a shared Drive folder.

    Files with the same name in the destination folder are updated in place;
    this does not delete remote files absent from the local source.
    """
    from googleapiclient.http import MediaFileUpload

    root = Path(source).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    uploaded: list[str] = []
    folder_ids: dict[Path, str] = {root: parent_folder_id}

    for directory in [root, *sorted(p for p in root.rglob("*") if p.is_dir())]:
        if directory != root:
            folder_ids[directory] = ensure_folder(
                service, directory.name, folder_ids[directory.parent]
            )
        remote_parent = folder_ids[directory]
        for path in sorted(p for p in directory.iterdir() if p.is_file()):
            escaped = path.name.replace("'", "\\'")
            matches = service.files().list(
                q=f"'{remote_parent}' in parents and name = '{escaped}' and trashed = false",
                fields="files(id)",
                pageSize=100,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ).execute().get("files", [])
            media = MediaFileUpload(
                str(path), mimetype=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                resumable=True,
            )
            if matches:
                result = service.files().update(
                    fileId=matches[0]["id"], media_body=media,
                    supportsAllDrives=True, fields="id",
                ).execute()
            else:
                result = service.files().create(
                    body={"name": path.name, "parents": [remote_parent]},
                    media_body=media, supportsAllDrives=True, fields="id",
                ).execute()
            uploaded.append(result["id"])
    return uploaded
