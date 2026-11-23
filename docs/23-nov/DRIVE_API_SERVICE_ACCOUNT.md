# Drive API via service account

The notebook's Step 2 now has an opt-in `service_account` mode. It does not
mount Drive; it creates a Drive API client using a service-account JSON key
stored in Colab Secrets and verifies its `client_email` against
`baliweb-gsc@concrete-crow-424203-p0.iam.gserviceaccount.com`.

## One-time setup

1. In Google Cloud Console, create/download a JSON key for the service account.
   Do not commit the key or paste it into a notebook cell.
2. In Colab, open **Secrets** and add `GDRIVE_SERVICE_ACCOUNT_JSON`; its value
   must be the complete JSON key text. Enable notebook access for that secret.
3. Share the target Drive folder with the service-account email as Editor.
4. Set `DRIVE_ACCESS_MODE = 'service_account'` in Step 2 and run it.

**Storage caveat:** service accounts do not have personal Drive storage quota.
For reliable API uploads, the destination should be in a Google Shared Drive
where the service account is a member. Uploads into an ordinary My Drive shared
folder can fail with a quota/ownership error even when the account is Editor.
If that happens, use a Shared Drive destination or the regular Drive mount.

The shared folder ID is configured as
`1Wx3jxA0RRsLnPkbhmt_Na8jRCrjYHDaA`. The Drive helper exposes:

```python
download_folder(DRIVE_API, folder_id, '/content/local_copy')
remote_folder = ensure_folder(DRIVE_API, 'results', DRIVE_SHARED_FOLDER_ID)
upload_folder(DRIVE_API, '/content/local_results', remote_folder)
```

Downloads recursively mirror a folder. Uploads create/update files beneath the
destination and do not delete remote files. API access is an explicit
download/upload operation, not a mounted filesystem: callers must choose the
specific input folder and output directory. Large training checkpoints should
be synced deliberately (for example after each completed run); a runtime reset
before upload loses local-only changes.

Existing path-based notebook steps retain their mount-based defaults. Selecting
service-account mode only initializes the API client; it does not silently
rewrite every legacy input/output path or automatically copy large datasets.
Steps 31 and 33 stage their required 512-data folders locally and upload their
separate result trees. Step 32 uploads its audit output to a `face_audit_v1`
child folder. Step 31 can locate the source run by its exact folder name below
the shared root; set `DRIVE_SOURCE_RUN_FOLDER_ID` if that name is ambiguous or
the source run is shared separately. Step 33 can similarly use
`DRIVE_SKIN_FILTER_RUN_FOLDER_ID` to resume a specific remote filtered run.

Other legacy steps (including Step 31b training) retain their existing
mount-based paths. This avoids silently downloading large training checkpoints
or changing the behavior of unrelated workflows. API download/upload is not a
live filesystem mount: an interrupted operation must be rerun to sync outputs.
