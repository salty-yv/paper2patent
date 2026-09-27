from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import uvicorn

from patent_writer.web import app


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8765)
