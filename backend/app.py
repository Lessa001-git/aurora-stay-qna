
# Import necessary libraries
import logging

from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Import RAG utility functions
from rag_utils import retrieve, rag


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

EMBED_MODEL = "HarishMaths/Hotel-Policy-Embedding"
VECTORSTORE_PATH = "faiss_index"

# Validation limits (defaults match the evaluated configuration)
ALLOWED_MODELS = {"gpt-4o-mini"}
DEFAULT_K = 4
MAX_QUERY_CHARS = 2000

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("aurora-api")


# ---------------------------------------------------------
# Initialize the Flask application
# ---------------------------------------------------------

rag_api = Flask("Aurora Stay API")


# ---------------------------------------------------------
# Load the embedding model
# ---------------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


# ---------------------------------------------------------
# Load the saved FAISS vector store
# ---------------------------------------------------------

vectorstore = FAISS.load_local(
    VECTORSTORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

print(f"Vectors stored: {vectorstore.index.ntotal}")


# ---------------------------------------------------------
# Input validation and error responses
# ---------------------------------------------------------

class BadRequest(ValueError):
    """Raised when the request body fails validation."""


def error(status, message):
    """Return a JSON error body with the given HTTP status."""
    return jsonify({"error": message, "status": status}), status


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def parse_request(data, need_generation):
    """Validate the JSON body and return clean parameters, or raise BadRequest."""
    if not isinstance(data, dict):
        raise BadRequest("Request body must be a JSON object.")

    query = data.get("query")
    if not isinstance(query, str) or not query.strip():
        raise BadRequest("'query' is required and must be a non-empty string.")
    if len(query) > MAX_QUERY_CHARS:
        raise BadRequest(f"'query' must be {MAX_QUERY_CHARS} characters or fewer.")

    k = data.get("k", DEFAULT_K)
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 10:
        raise BadRequest("'k' must be an integer between 1 and 10.")

    params = {"query": query.strip(), "k": k}

    if need_generation:
        model_name = data.get("model_name", "gpt-4o-mini")
        if model_name not in ALLOWED_MODELS:
            raise BadRequest(f"'model_name' must be one of {sorted(ALLOWED_MODELS)}.")

        temperature = data.get("temperature", 0.0)
        if not _is_number(temperature) or not 0 <= temperature <= 1:
            raise BadRequest("'temperature' must be a number between 0 and 1.")

        top_p = data.get("top_p", 1.0)
        if not _is_number(top_p) or not 0 <= top_p <= 1:
            raise BadRequest("'top_p' must be a number between 0 and 1.")

        max_tokens = data.get("max_tokens", 512)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= 2048:
            raise BadRequest("'max_tokens' must be an integer between 1 and 2048.")

        params.update(
            model_name=model_name,
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=max_tokens,
        )

    return params


@rag_api.errorhandler(HTTPException)
def handle_http_error(e):
    """Return Flask's own errors (404, 405, ...) as JSON instead of HTML."""
    return error(e.code, e.description)


@rag_api.errorhandler(Exception)
def handle_unexpected_error(e):
    """Catch anything unhandled, log it, and return a JSON 500."""
    log.exception("Unhandled error")
    return error(500, "Internal server error.")


def run_rag(p):
    """Run the unchanged RAG pipeline with validated parameters."""
    return rag(
        query=p["query"],
        k=p["k"],
        model_name=p["model_name"],
        temperature=p["temperature"],
        top_p=p["top_p"],
        max_tokens=p["max_tokens"],
        vectorstore=vectorstore
    )


# ---------------------------------------------------------
# Default route
# ---------------------------------------------------------

@rag_api.get("/")
def home():
    """
    Handles GET requests to the root URL.
    """
    return "Welcome to the Aurora Stay RAG API!"


# ---------------------------------------------------------
# Relevant chunks endpoint
# ---------------------------------------------------------

@rag_api.post("/v1/relevant_chunks")
def relevant_chunks():
    """
    Handles POST requests to /v1/relevant_chunks.

    Example request:
    {
        "query": "What is the cancellation policy?",
        "k": 4
    }

    Returns the top-k relevant chunks, 400 for invalid input,
    or 503 if retrieval fails.
    """
    try:
        p = parse_request(request.get_json(silent=True), need_generation=False)
    except BadRequest as e:
        return error(400, str(e))

    try:
        chunks = retrieve(query=p["query"], k=p["k"], vectorstore=vectorstore)
    except Exception:
        log.exception("Retrieval failed")
        return error(503, "Retrieval is temporarily unavailable. Please try again.")

    return jsonify({
        "query": p["query"],
        "k": p["k"],
        "relevant_chunks": chunks
    })


# ---------------------------------------------------------
# Answer with relevant chunks endpoint
# ---------------------------------------------------------

@rag_api.post("/v1/answer_with_relevant_chunks")
def answer_with_relevant_chunks():
    """
    Handles POST requests to /v1/answer_with_relevant_chunks.

    Example request:
    {
        "query": "What is the cancellation policy?",
        "k": 4,
        "model_name": "gpt-4o-mini",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 512
    }

    Returns the answer and the chunks used, 400 for invalid input,
    or 502 if the language model call fails.
    """
    try:
        p = parse_request(request.get_json(silent=True), need_generation=True)
    except BadRequest as e:
        return error(400, str(e))

    try:
        answer, chunks = run_rag(p)
    except Exception:
        log.exception("Generation failed")
        return error(502, "The language model did not respond. Please try again or look up the policy manually.")

    return jsonify({
        "query": p["query"],
        "k": p["k"],
        "model_name": p["model_name"],
        "temperature": p["temperature"],
        "top_p": p["top_p"],
        "max_tokens": p["max_tokens"],
        "relevant_chunks": chunks,
        "answer": answer
    })


# ---------------------------------------------------------
# Answer endpoint
# ---------------------------------------------------------

@rag_api.post("/v1/answer")
def answer():
    """
    Handles POST requests to /v1/answer.

    Example request:
    {
        "query": "What is the cancellation policy?",
        "k": 4,
        "model_name": "gpt-4o-mini",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 512
    }

    Returns only the generated answer, 400 for invalid input,
    or 502 if the language model call fails.
    """
    try:
        p = parse_request(request.get_json(silent=True), need_generation=True)
    except BadRequest as e:
        return error(400, str(e))

    try:
        generated_answer, _ = run_rag(p)
    except Exception:
        log.exception("Generation failed")
        return error(502, "The language model did not respond. Please try again or look up the policy manually.")

    return jsonify({
        "answer": generated_answer
    })


# ---------------------------------------------------------
# Run the Flask application
# ---------------------------------------------------------

if __name__ == "__main__":
    rag_api.run(debug=True)
