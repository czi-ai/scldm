#!/usr/bin/env -S python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "boto3",
# ]
# ///
#
# ruff: noqa: BLE001
#
import argparse
import collections
import dataclasses
import hashlib
import pathlib
import sys
from typing import Literal

import boto3
from botocore import UNSIGNED
from botocore.config import Config

ALL_GROUPS = "all"

DEFAULT_DESTINATION = pathlib.Path(__file__).parent.resolve() / "_artifacts"


@dataclasses.dataclass
class Artifact:
    remote_uri: str
    local_path: str
    groups: set[str] | Literal["all"] = ALL_GROUPS


@dataclasses.dataclass
class AWSCredentials:
    profile: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None
    use_environment_credentials: bool = False


ARTIFACTS = [
    Artifact("s3://czi-scldm/datasets/dentategyrus_test.h5ad", "datasets/dentategyrus_test.h5ad", groups={"datasets"}),
    Artifact(
        "s3://czi-scldm/datasets/dentategyrus_train.h5ad", "datasets/dentategyrus_train.h5ad", groups={"datasets"}
    ),
    Artifact(
        "s3://czi-scldm/fm_observational/dentate_gyrus.ckpt",
        "fm_observational/dentate_gyrus.ckpt",
        groups={"fm_observational"},
    ),
    Artifact(
        "s3://czi-scldm/fm_observational/dentate_gyrus.yaml",
        "fm_observational/dentate_gyrus.yaml",
        groups={"fm_observational"},
    ),
    Artifact(
        "s3://czi-scldm/fm_observational/dentate_gyrus_log_size_factor_mu.pkl",
        "fm_observational/dentate_gyrus_log_size_factor_mu.pkl",
        groups={"fm_observational"},
    ),
    Artifact(
        "s3://czi-scldm/fm_observational/dentate_gyrus_log_size_factor_sd.pkl",
        "fm_observational/dentate_gyrus_log_size_factor_sd.pkl",
        groups={"fm_observational"},
    ),
    Artifact("s3://czi-scldm/fm_observational/hlca.ckpt", "fm_observational/hlca.ckpt", groups={"fm_observational"}),
    Artifact("s3://czi-scldm/fm_observational/hlca.yaml", "fm_observational/hlca.yaml", groups={"fm_observational"}),
    Artifact(
        "s3://czi-scldm/fm_observational/hlca_log_size_factor_mu.pkl",
        "fm_observational/hlca_log_size_factor_mu.pkl",
        groups={"fm_observational"},
    ),
    Artifact(
        "s3://czi-scldm/fm_observational/hlca_log_size_factor_sd.pkl",
        "fm_observational/hlca_log_size_factor_sd.pkl",
        groups={"fm_observational"},
    ),
    Artifact(
        "s3://czi-scldm/fm_observational/tabula_muris.ckpt",
        "fm_observational/tabula_muris.ckpt",
        groups={"fm_observational"},
    ),
    Artifact(
        "s3://czi-scldm/fm_observational/tabula_muris.yaml",
        "fm_observational/tabula_muris.yaml",
        groups={"fm_observational"},
    ),
    Artifact(
        "s3://czi-scldm/fm_observational/tabula_muris_log_size_factor_mu.pkl",
        "fm_observational/tabula_muris_log_size_factor_mu.pkl",
        groups={"fm_observational"},
    ),
    Artifact(
        "s3://czi-scldm/fm_observational/tabula_muris_log_size_factor_sd.pkl",
        "fm_observational/tabula_muris_log_size_factor_sd.pkl",
        groups={"fm_observational"},
    ),
    Artifact(
        "s3://czi-scldm/fm_perturbation/czb_cd4_naive_holdout_log_size_factor_mu.pkl",
        "fm_perturbation/czb_cd4_naive_holdout_log_size_factor_mu.pkl",
        groups={"fm_perturbation"},
    ),
    Artifact(
        "s3://czi-scldm/fm_perturbation/czb_cd4_naive_holdout_log_size_factor_sd.pkl",
        "fm_perturbation/czb_cd4_naive_holdout_log_size_factor_sd.pkl",
        groups={"fm_perturbation"},
    ),
    Artifact("s3://czi-scldm/fm_perturbation/parse1m.ckpt", "fm_perturbation/parse1m.ckpt", groups={"fm_perturbation"}),
    Artifact("s3://czi-scldm/fm_perturbation/parse1m.yaml", "fm_perturbation/parse1m.yaml", groups={"fm_perturbation"}),
    Artifact(
        "s3://czi-scldm/fm_perturbation/replogle.ckpt", "fm_perturbation/replogle.ckpt", groups={"fm_perturbation"}
    ),
    Artifact(
        "s3://czi-scldm/fm_perturbation/replogle.yaml", "fm_perturbation/replogle.yaml", groups={"fm_perturbation"}
    ),
    Artifact(
        "s3://czi-scldm/fm_perturbation/replogle_log_size_factor_mu.pkl",
        "fm_perturbation/replogle_log_size_factor_mu.pkl",
        groups={"fm_perturbation"},
    ),
    Artifact(
        "s3://czi-scldm/fm_perturbation/replogle_log_size_factor_sd.pkl",
        "fm_perturbation/replogle_log_size_factor_sd.pkl",
        groups={"fm_perturbation"},
    ),
    Artifact("s3://czi-scldm/vae_census/20M.ckpt", "vae_census/20M.ckpt", groups={"vae_census"}),
    Artifact("s3://czi-scldm/vae_census/20M.yaml", "vae_census/20M.yaml", groups={"vae_census"}),
    Artifact("s3://czi-scldm/vae_census/270M.ckpt", "vae_census/270M.ckpt", groups={"vae_census"}),
    Artifact("s3://czi-scldm/vae_census/270M.yaml", "vae_census/270M.yaml", groups={"vae_census"}),
    Artifact("s3://czi-scldm/vae_census/70M.ckpt", "vae_census/70M.ckpt", groups={"vae_census"}),
    Artifact("s3://czi-scldm/vae_census/70M.yaml", "vae_census/70M.yaml", groups={"vae_census"}),
    Artifact(
        "s3://czi-scldm/vae_census/concatenated_unique_genes.parquet",
        "vae_census/concatenated_unique_genes.parquet",
        groups={"vae_census"},
    ),
]


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse s3://bucket/key/path into (bucket, key)"""
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri}")

    parts = uri.removeprefix("s3://").split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


def _does_local_file_match_etag(local_path: pathlib.Path, etag: str) -> bool:
    """
    Returns True if the local file's md5 hash matches the ETag

    This does not properly handle multi-part uploads
    """
    etag = etag.strip('"')

    # multipart upload ETags contains a hyphen
    if "-" in etag:
        # Skip verifying the hash because the process is complicated
        # assume if there's a file there, it's the correct one
        return local_path.is_file()

    # simple single part case, the ETag is the md5 hash
    md5_hash = hashlib.md5()
    with open(local_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5_hash.update(chunk)

    local_md5 = md5_hash.hexdigest()
    return local_md5 == etag


def _download_file(client, bucket: str, key: str, local_path: pathlib.Path):
    """
    Download a single file from S3 to local path.

    Checks ETag first if file exists and skips download if it matches.
    """
    if local_path.is_file():
        try:
            head_response = client.head_object(Bucket=bucket, Key=key)
            etag = head_response["ETag"]
            if _does_local_file_match_etag(local_path, etag):
                return  # it's already been downloaded
        except Exception:
            pass  # we might still be able to download it even though we couldn't check the etag

    local_path.parent.mkdir(parents=True, exist_ok=True)

    client.download_file(bucket, key, str(local_path))


def download_from_s3(client, destination: pathlib.Path, groups: set[str] | Literal["all"] = ALL_GROUPS) -> list[str]:
    """
    Download files from S3 to local.

    Returns list of error messages encountered during download.
    """
    destination.mkdir(parents=True, exist_ok=True)
    errors = []

    for artifact in ARTIFACTS:
        # first check to see if we're downloading everything, or if the artifact should always be downloaded
        if groups != ALL_GROUPS and artifact.groups != ALL_GROUPS:
            # if not, see if there's overlap between the groups we're downloading and the artifact's groups
            if not groups.intersection(artifact.groups):
                continue  # no common groups and we're not downloading all, so skip it
        try:
            bucket, key = _parse_s3_uri(artifact.remote_uri)

            is_prefix = artifact.remote_uri.endswith("/")

            if is_prefix:
                # download everything under the prefix
                paginator = client.get_paginator("list_objects_v2")
                pages = paginator.paginate(Bucket=bucket, Prefix=key)

                for page in pages:
                    if "Contents" not in page:
                        continue

                    for obj in page["Contents"]:
                        obj_key = obj["Key"]
                        relative_path = obj_key[len(key) :]
                        if not relative_path or relative_path.endswith("/"):
                            continue  # skip directory markers
                        local_path = destination / artifact.local_path / relative_path
                        try:
                            _download_file(client, bucket, obj_key, local_path)
                        except Exception as e:
                            errors.append(f"Error downloading {artifact.remote_uri}{relative_path}: {e}")
            else:
                # download the single file
                local_path = destination / artifact.local_path
                try:
                    _download_file(client, bucket, key, local_path)
                except Exception as e:
                    errors.append(f"Error downloading {artifact.remote_uri}: {e}")

        except Exception as e:
            errors.append(f"Error processing artifact {artifact.remote_uri}: {e}")

    return errors


class ValidationError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _validate():
    """Check that the artifact destinations make sense"""
    local_path_count = collections.Counter([artifact.local_path for artifact in ARTIFACTS])
    multiples = {filename for filename, count in local_path_count.items() if count > 1}
    if multiples:
        raise ValidationError(f"Multiple artifacts will be downloaded to the same local path: {multiples!r}")

    for artifact in ARTIFACTS:
        if artifact.groups != ALL_GROUPS and not artifact.groups:
            raise ValidationError(f"Artifact {artifact.remote_uri} isn't assigned to any download groups")


class DownloadError(Exception):
    def __init__(self, messages: list[str]):
        super().__init__(messages)
        self.messages = messages


def download_artifacts(
    destination: pathlib.Path | str = DEFAULT_DESTINATION,
    credentials: AWSCredentials | None = None,
    groups: set[str] | Literal["all"] = ALL_GROUPS,
):
    """
    Download artifacts in the specified groups, optionally using supplied AWS Credentials.

    Returns list of error messages encountered during download.
    """
    _validate()

    destination = pathlib.Path(destination)

    credentials = credentials or AWSCredentials()  # simplifies the logic if this is not None

    if credentials.profile:
        # use that profile
        session = boto3.Session(profile_name=credentials.profile)
        client = session.client("s3")
    else:
        # try to use what credentials they provided
        client_kwargs = {}
        if credentials.aws_access_key_id:
            client_kwargs["aws_access_key_id"] = credentials.aws_access_key_id
        if credentials.aws_secret_access_key:
            client_kwargs["aws_secret_access_key"] = credentials.aws_secret_access_key
        if credentials.aws_session_token:
            client_kwargs["aws_session_token"] = credentials.aws_session_token

        # If no credentials provided and not having boto3 get them from environment, use unsigned requests
        if (not credentials.use_environment_credentials) and not any(
            [credentials.aws_access_key_id, credentials.aws_secret_access_key, credentials.aws_session_token]
        ):
            client_kwargs["config"] = Config(signature_version=UNSIGNED)

        client = boto3.client("s3", **client_kwargs)

    errors = download_from_s3(client, destination, groups)
    if errors:
        raise DownloadError(errors)


def main():
    parser = argparse.ArgumentParser(description="Download artifacts from S3 to local filesystem")
    parser.add_argument(
        "--destination",
        type=pathlib.Path,
        default=DEFAULT_DESTINATION,
        help=f"Destination directory (default: {DEFAULT_DESTINATION})",
    )
    parser.add_argument(
        "--group",
        action="append",
        dest="groups",
        help="Group(s) to download (can be specified multiple times, or comma-separated). Use 'all' to download all artifacts (default: all).",
    )
    parser.add_argument("--profile", help="AWS profile name to use")
    parser.add_argument("--aws-access-key-id", help="AWS access key ID (if no profile specified)")
    parser.add_argument(
        "--aws-secret-access-key",
        help="AWS secret access key (if no profile specified)",
    )
    parser.add_argument(
        "--aws-session-token",
        help="AWS session token (optional, if no profile specified)",
    )
    parser.add_argument(
        "--aws-use-environment-credentials",
        action="store_true",
        help="Use AWS credentials from environment/config files instead of unsigned requests",
    )

    args = parser.parse_args()

    # specifying a profile overrides any of these other credentials
    if args.profile and any(
        [
            args.aws_access_key_id,
            args.aws_secret_access_key,
            args.aws_session_token,
        ]
    ):
        parser.error(
            "Cannot specify both --profile and explicit AWS credentials "
            "(--aws-access-key-id, --aws-secret-access-key, --aws-session-token)"
        )

    groups: set[str] | Literal["all"] = ALL_GROUPS
    if args.groups:
        # handle comma-separated groups
        all_group_values = []
        for group_arg in args.groups:
            all_group_values.extend(g.strip() for g in group_arg.split(","))

        if ALL_GROUPS in all_group_values:
            groups = ALL_GROUPS
        else:
            groups = set(all_group_values)

    credentials = AWSCredentials(
        profile=args.profile,
        aws_access_key_id=args.aws_access_key_id,
        aws_secret_access_key=args.aws_secret_access_key,
        aws_session_token=args.aws_session_token,
        use_environment_credentials=args.aws_use_environment_credentials,
    )

    try:
        errors = download_artifacts(destination=args.destination, credentials=credentials, groups=groups)
    except ValidationError as e:
        print("ABORTING DOWNLOAD", file=sys.stderr)
        print(e.message, file=sys.stderr)
        sys.exit(1)
    except DownloadError as e:
        print("\nErrors encountered during download:", file=sys.stderr)
        for error in e.messages:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
