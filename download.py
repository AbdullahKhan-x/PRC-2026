import json
import os
import boto3

creds = json.load(open("credentials.json"))
key = creds["accessKey"]
secret = creds["secretKey"]

s3 = boto3.client(
    "s3",
    endpoint_url="https://s3.opensky-network.org",
    aws_access_key_id=key,
    aws_secret_access_key=secret,
)

os.makedirs("data/raw", exist_ok=True)

objects = s3.list_objects_v2(Bucket="prc-2026-datasets")["Contents"]

for obj in objects:
    name = obj["Key"]
    dest = os.path.join("data/raw", name)
    if os.path.exists(dest):
        print(f"skip {name}")
        continue
    print(f"downloading {name} ({obj['Size'] / 1e6:.1f} MB)")
    s3.download_file("prc-2026-datasets", name, dest)

print("done")
