import io
import json
import zipfile

import requests
import frontmatter


def read_repo_data(repo_owner, repo_name, branch="main"):
    """Download and parse all markdown files from a GitHub repository."""
    prefix = 'https://codeload.github.com'
    url = f'{prefix}/{repo_owner}/{repo_name}/zip/refs/heads/{branch}'

    print(f"Downloading {repo_owner}/{repo_name}...")
    resp = requests.get(url)

    if resp.status_code != 200:
        raise Exception(f"Failed to download repository: {resp.status_code}")

    repository_data = []
    zf = zipfile.ZipFile(io.BytesIO(resp.content))

    # Strip the top-level archive directory from filenames
    archive_prefix = f"{repo_name}-{branch}/"

    for file_info in zf.infolist():
        filename = file_info.filename
        if not (filename.lower().endswith('.md') or filename.lower().endswith('.mdx')):
            continue

        try:
            with zf.open(file_info) as f_in:
                raw = f_in.read().decode('utf-8', errors='ignore')
                post = frontmatter.loads(raw)
                data = post.to_dict()
                clean_name = filename.removeprefix(archive_prefix)
                data['filename'] = clean_name
                repository_data.append(data)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    zf.close()
    return repository_data


def sliding_window(seq, size, step):
    """Create sliding windows over a string."""
    result = []
    for i in range(0, len(seq), step):
        chunk = seq[i:i + size]
        result.append({'start': i, 'chunk': chunk})
        if i + size >= len(seq):
            break
    return result


def chunk_documents(docs, size=2000, step=1000):
    """Chunk documents using a sliding window approach."""
    chunks = []
    for doc in docs:
        doc_copy = doc.copy()
        content = doc_copy.pop('content', '')
        if not content:
            continue
        windows = sliding_window(content, size, step)
        for w in windows:
            w.update(doc_copy)
        chunks.extend(windows)
    return chunks


def index_data(repo_owner, repo_name, branch="main", chunk=True, size=2000, step=1000):
    """Download repo, parse docs, optionally chunk, and return documents."""
    docs = read_repo_data(repo_owner, repo_name, branch=branch)
    print(f"Downloaded {len(docs)} documents")

    if chunk:
        chunks = chunk_documents(docs, size=size, step=step)
        print(f"Created {len(chunks)} chunks")
        return chunks
    return docs
