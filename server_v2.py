# MS2 Spectral Search REST API using Flask, MatchMS, and MSEntropy
#
# This script creates a web server that provides a RESTful interface
# for searching MS2 spectra against MSP libraries. It now includes:
# - Flash Entropy Search, MatchMS Cosine, and a combined hybrid search
# - Optional precursor m/z for EI-MS support
# - Library uploading with duplicate-checking
# - API usage tracking for grant reporting
# - A simple web GUI with library upload and parameter overrides
#
# To run this application:
# 1. Make sure you have Python installed.
# 2. Install the required libraries:
#    pip install Flask matchms ms_entropy
# 3. Create a folder named `libraries` and `logs` in the same directory as this script.
# 4. Place your MSP spectral library files (e.g., `my_lib.msp`) inside the `libraries` folder.
# 5. Run this script: python server.py
# 6. The server will start on http://0.0.0.0:5000.

from flask import Flask, request, jsonify, render_template_string
import matchms
from matchms import Spectrum
from matchms.similarity import CosineGreedy
from matchms.filtering import normalize_intensities, require_minimum_number_of_peaks
import ms_entropy
import numpy as np
import os
import json
import functools
from typing import List, Dict, Any, Union, Optional
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import hashlib
import shutil
from werkzeug.utils import secure_filename

# --- Configuration ---
# The folder where your .msp library files are stored.
LIBRARY_FOLDER = './libraries'
# The folder for log files.
LOGS_FOLDER = './logs'
# Log file name for API usage.
USAGE_LOG_FILE = os.path.join(LOGS_FOLDER, 'api_usage.log')
# Precursor m/z tolerance for finding candidate spectra (only used if precursor is present).
PRECURSOR_MZ_TOLERANCE = 0.5
# Fragment m/z tolerance for aligning peaks in spectra.
FRAGMENT_MZ_TOLERANCE = 0.2
# Minimum number of matched peaks required to consider a match.
MIN_MATCHED_PEAKS = 6
# Maximum number of libraries to keep in the cache.
MAX_CACHED_LIBRARIES = 10
# Number of candidates to pass from Flash Entropy to Cosine for combined search.
COMBINED_SEARCH_CANDIDATES = 100

# --- Flask Application Setup ---
app = Flask(__name__)
# Use the custom JSON encoder to handle numpy types
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
app.json_encoder = NpEncoder

# --- Logging Setup for Usage Tracking ---
if not os.path.exists(LOGS_FOLDER):
    os.makedirs(LOGS_FOLDER)

# Configure the API usage logger
usage_logger = logging.getLogger('api_usage_logger')
usage_logger.setLevel(logging.INFO)
# Use RotatingFileHandler to manage log file size and rotation
file_handler = RotatingFileHandler(USAGE_LOG_FILE, maxBytes=1024 * 1024, backupCount=5)
# Set a custom format for the log entries
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
usage_logger.addHandler(file_handler)

def log_usage(endpoint: str, method: str, library: str, query_count: int, metadata_filter: Optional[Dict[str, Any]] = None):
    """Logs API usage details for grant reporting."""
    timestamp = datetime.now().isoformat()
    ip_address = request.remote_addr
    filter_str = json.dumps(metadata_filter) if metadata_filter else "None"
    log_message = f"IP: {ip_address}, Endpoint: {endpoint}, Method: {method}, Library: {library}, QueryCount: {query_count}, MetadataFilter: {filter_str}"
    usage_logger.info(log_message)


# --- Library Loading and Processing (Lazy) ---
def process_spectrum(s: Spectrum) -> Union[Spectrum, None]:
    """
    Applies a standard set of processing filters to a matchms Spectrum object.
    This ensures that library and query spectra are treated the same way.
    """
    if s is None:
        return None
    s = normalize_intensities(s)
    s = require_minimum_number_of_peaks(s, n_required=1)
    return s

@functools.lru_cache(maxsize=MAX_CACHED_LIBRARIES)
def load_and_process_library(library_name: str) -> List[Spectrum]:
    """
    Loads and processes a single MSP library file and caches the result.
    The library_name should be the filename without the .msp extension.
    """
    filepath = os.path.abspath(os.path.join(LIBRARY_FOLDER, f"{library_name}.msp"))
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Library file not found: {filepath}")

    # load_from_msp returns a generator, so we convert it to a list
    spectra_generator = matchms.importing.load_from_msp(filepath)
    
    processed_spectra = []
    for spec in spectra_generator:
        try:
            processed_spec = process_spectrum(spec)
            if processed_spec is not None:
                processed_spectra.append(processed_spec)
        except Exception as e:
            print(f"Error processing spectrum in {library_name}.msp: {e}")

    print(f"Successfully loaded and processed library: {library_name}.msp with {len(processed_spectra)} spectra.")
    return processed_spectra

def get_available_libraries() -> List[str]:
    """
    Returns a list of available library names (without .msp extension).
    """
    if not os.path.exists(LIBRARY_FOLDER):
        os.makedirs(LIBRARY_FOLDER)
        return []
    return [os.path.splitext(f)[0] for f in os.listdir(LIBRARY_FOLDER) if f.lower().endswith(".msp")]

# --- Library Upload Checksum Utilities ---
def get_file_checksum(file_path: str) -> str:
    """Calculates the SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        # Read in 8KB chunks
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def get_existing_library_checksums() -> set:
    """Gets the checksums of all .msp files in the library folder."""
    if not os.path.exists(LIBRARY_FOLDER):
        os.makedirs(LIBRARY_FOLDER)
        return set()
    
    checksums = set()
    for f in os.listdir(LIBRARY_FOLDER):
        if f.lower().endswith(".msp"):
            file_path = os.path.join(LIBRARY_FOLDER, f)
            checksums.add(get_file_checksum(file_path))
    return checksums

# --- Spectrum Matching Logic ---
def search_spectrum_with_matchms(
    query_spectrum: Spectrum, 
    library_spectra: List[Spectrum], 
    precursor_mz_tolerance: float, 
    fragment_tolerance: float, 
    min_matched_peaks: int
) -> tuple[Union[Spectrum, None], float]:
    """
    Searches a single query spectrum against a given library using matchms CosineGreedy.
    If query_spectrum has no precursor_mz, it searches the entire library.
    """
    best_match = None
    best_score = 0.0
    
    query_precursor_mz = query_spectrum.get('precursor_mz')

    # 1. Filter library by precursor m/z to get candidate spectra
    #    If query has no precursor_mz (e.g., EI-MS), compare against the whole library.
    if query_precursor_mz is not None:
        candidate_spectra = [
            spec for spec in library_spectra 
            if abs(spec.get('precursor_mz', 0) - query_precursor_mz) <= precursor_mz_tolerance
        ]
    else:
        # No query precursor, so search all library spectra
        candidate_spectra = library_spectra
    
    if not candidate_spectra:
        return None, 0.0

    # 2. Calculate similarity using matchms CosineGreedy
    cosine_greedy = CosineGreedy(tolerance=fragment_tolerance)
    scores = cosine_greedy.matrix([query_spectrum], candidate_spectra)

    # 3. Find the best match
    if scores.shape[1] > 0:
        best_match_idx = np.argmax(scores[0, :])
        score_tuple = scores[0, best_match_idx] # (score, matched_peaks)
        
        # Check if the match meets the minimum peak requirement
        if score_tuple[1] >= min_matched_peaks:
            best_score = score_tuple[0]
            best_match = candidate_spectra[best_match_idx]

    return best_match, best_score

def search_spectrum_with_entropy(
    query_spectrum: Spectrum, 
    library_spectra: List[Spectrum],
    precursor_mz_tolerance: float
) -> tuple[Union[Spectrum, None], float]:
    """
    Searches a single query spectrum against a given library using Flash Entropy Search.
    If query_spectrum has no precursor_mz, the precursor tolerance is not used.
    """
    query_precursor_mz = query_spectrum.get('precursor_mz')
    library_metadata = [spec.metadata for spec in library_spectra]

    # Convert matchms spectrum objects to the format expected by ms-entropy
    query_data = {'precursor_mz': query_precursor_mz, 'peaks': list(zip(query_spectrum.mz, query_spectrum.intensities))}
    library_data = [{'precursor_mz': lib.get('precursor_mz'), 'peaks': list(zip(lib.mz, lib.intensities))} for lib in library_spectra]
    
    # Only apply precursor tolerance if the query has a precursor m/z
    search_args = {}
    if query_precursor_mz is not None:
        search_args['precursor_mz_tolerance'] = precursor_mz_tolerance

    # Perform the search
    search_results = ms_entropy.flash_entropy_search(
        query_data, 
        library_data, 
        **search_args
    )

    if not search_results:
        return None, 0.0

    # The result from ms-entropy is a list of tuples: (library_index, score)
    best_match_idx, best_score = search_results[0]
    
    # ms-entropy might return a score > 1.0, so we normalize it for consistency
    normalized_score = min(best_score, 1.0)

    best_match = library_spectra[best_match_idx]
    
    return best_match, normalized_score

def search_spectrum_with_combined(
    query_spectrum: Spectrum,
    library_spectra: List[Spectrum],
    precursor_mz_tolerance: float,
    fragment_tolerance: float, # <-- MODIFIED
    min_matched_peaks: int,     # <-- MODIFIED
    top_n_candidates: int
) -> tuple[Union[Spectrum, None], float]:
    """
    Performs a two-stage search:
    1. Initial fast filter with Flash Entropy to get top N candidates.
    2. Refine the search with CosineGreedy on the candidates.
    """
    # Stage 1: Fast filtering with Flash Entropy to get top N candidates
    query_precursor_mz = query_spectrum.get('precursor_mz')
    query_data = {'precursor_mz': query_precursor_mz, 'peaks': list(zip(query_spectrum.mz, query_spectrum.intensities))}
    library_data = [{'precursor_mz': lib.get('precursor_mz'), 'peaks': list(zip(lib.mz, lib.intensities))} for lib in library_spectra]
    
    # Only apply precursor tolerance if the query has a precursor m/z
    search_args = {}
    if query_precursor_mz is not None:
        search_args['precursor_mz_tolerance'] = precursor_mz_tolerance
    
    # Get top N search results from Flash Entropy
    entropy_results = ms_entropy.flash_entropy_search(
        query_data,
        library_data,
        **search_args
    )

    if not entropy_results:
        return None, 0.0

    # Extract the top N candidates and their indices
    candidate_indices = [idx for idx, _ in entropy_results[:top_n_candidates]]
    candidate_spectra = [library_spectra[idx] for idx in candidate_indices]

    if not candidate_spectra:
        return None, 0.0

    # Stage 2: High-fidelity refinement with Matchms CosineGreedy
    cosine_greedy = CosineGreedy(tolerance=fragment_tolerance) # <-- MODIFIED
    scores = cosine_greedy.matrix([query_spectrum], candidate_spectra)
    
    best_match = None
    best_score = 0.0

    if scores.shape[1] > 0:
        best_match_idx = np.argmax(scores[0, :])
        score_tuple = scores[0, best_match_idx]
        
        # Check if the match meets the minimum peak requirement
        if score_tuple[1] >= min_matched_peaks: # <-- MODIFIED
            best_score = score_tuple[0]
            best_match = candidate_spectra[best_match_idx]
    
    return best_match, best_score

# --- General Search Function ---
def perform_search(
    query_spectrum: Spectrum, 
    library_name: str,
    search_method: str = 'matchms_cosine',
    metadata_filter: Optional[Dict[str, Any]] = None,
    # --- MODIFIED: Added search parameters with defaults ---
    precursor_tol: float = PRECURSOR_MZ_TOLERANCE,
    fragment_tol: float = FRAGMENT_MZ_TOLERANCE,
    min_peaks: int = MIN_MATCHED_PEAKS,
    top_n: int = COMBINED_SEARCH_CANDIDATES
) -> tuple[Dict[str, Any], int]:
    """
    Performs a search on a single spectrum against a given library, with optional metadata filtering.
    Returns the result and HTTP status code.
    """
    
    # Extract query metadata to return with the result
    query_info = {
        k: query_spectrum.get(k) for k in ['name', 'scan_number', 'retention_time', 'precursor_mz'] 
        if query_spectrum.get(k) is not None
    }

    try:
        if library_name not in get_available_libraries():
            return {'error': f"Library '{library_name}' not found. Available libraries: {get_available_libraries()}"}, 404

        lib_spectra = load_and_process_library(library_name)

        # Apply metadata filter if provided
        if metadata_filter:
            filtered_lib_spectra = [
                s for s in lib_spectra if all(
                    str(s.get(k)) == str(v) for k, v in metadata_filter.items() # Compare as strings for simplicity
                )
            ]
            if not filtered_lib_spectra:
                return {'status': 'no_match_found', 'query_info': query_info, 'reason': 'No library spectra match the provided metadata filter.'}, 200
        else:
            filtered_lib_spectra = lib_spectra

        processed_query_spectrum = process_spectrum(query_spectrum)
        if processed_query_spectrum is None:
            return {'status': 'no_match_found', 'query_info': query_info, 'reason': 'Query spectrum is empty after processing.'}, 200

        # --- MODIFIED: Use passed-in parameters ---
        if search_method == 'matchms_cosine':
            best_match, best_score = search_spectrum_with_matchms(
                processed_query_spectrum, 
                filtered_lib_spectra,
                precursor_tol,
                fragment_tol,
                min_peaks
            )
        elif search_method == 'flash_entropy':
             best_match, best_score = search_spectrum_with_entropy(
                processed_query_spectrum,
                filtered_lib_spectra,
                precursor_tol
            )
        elif search_method == 'combined':
            best_match, best_score = search_spectrum_with_combined(
                processed_query_spectrum,
                filtered_lib_spectra,
                precursor_tol,
                fragment_tol,
                min_peaks,
                top_n
            )
        else:
            return {'error': f"Invalid search method '{search_method}'. Choose 'matchms_cosine', 'flash_entropy', or 'combined'."}, 400
            
        # --- MODIFIED: Add search params to result ---
        search_params_used = {
            'precursor_mz_tolerance': precursor_tol,
            'fragment_mz_tolerance': fragment_tol,
            'min_matched_peaks': min_peaks,
            'combined_search_candidates': top_n if search_method == 'combined' else None
        }

        if best_match:
            result = {
                'status': 'match_found',
                'query_info': query_info,
                'library': library_name,
                'search_method': search_method,
                'score': round(best_score, 4),
                'annotation': best_match.metadata,
                'search_params_used': search_params_used
            }
            return result, 200
        else:
            return {'status': 'no_matches', 'query_info': query_info, 'search_method': search_method, 'search_params_used': search_params_used}, 200
    
    except FileNotFoundError as e:
        return {'error': str(e)}, 404
    except Exception as e:
        app.logger.error(f"Unexpected error in perform_search: {e}", exc_info=True)
        return {'error': f"An unexpected error occurred: {e}"}, 500

# --- API Endpoints ---
@app.route('/libraries', methods=['GET'])
def get_libraries_endpoint():
    """
    An endpoint to list the names of all loaded libraries.
    """
    log_usage(request.endpoint, request.method, 'N/A', 1)
    libraries = get_available_libraries()
    if not libraries:
        return jsonify({"warning": "No libraries found. Please place .msp files in the 'libraries' folder."}), 200
    return jsonify({"available_libraries": libraries})

@app.route('/libraries/upload', methods=['POST'])
def upload_library_endpoint():
    """
    An endpoint to upload a new .msp library file.
    Checks for duplicates based on file content (SHA256 checksum).
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not file.filename.lower().endswith('.msp'):
        return jsonify({"error": "Invalid file type. Only .msp files are supported."}), 400

    log_usage(request.endpoint, request.method, file.filename, 1)

    try:
        # Save to a temporary location to calculate checksum
        safe_temp_name = secure_filename(f"temp_{file.filename}")
        temp_path = os.path.join(LOGS_FOLDER, safe_temp_name)
        file.save(temp_path)
        
        # Calculate checksum of new file
        new_checksum = get_file_checksum(temp_path)
        
        # Compare against existing library checksums
        existing_checksums = get_existing_library_checksums()
        
        if new_checksum in existing_checksums:
            os.remove(temp_path)
            return jsonify({"status": "duplicate", "message": "A library with this exact content already exists."}), 409
        else:
            # File is unique, move it to the library folder
            safe_filename = secure_filename(file.filename)
            final_path = os.path.join(LIBRARY_FOLDER, safe_filename)
            shutil.move(temp_path, final_path)
            
            # Clear the library cache to force a reload on next access
            load_and_process_library.cache_clear()
            
            return jsonify({"status": "success", "message": f"Library '{safe_filename}' uploaded successfully."}), 201

    except Exception as e:
        app.logger.error(f"Error during library upload: {e}", exc_info=True)
        # Clean up temp file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": f"An unexpected error occurred during upload: {e}"}), 500


@app.route('/search', methods=['POST'])
def search_endpoint():
    """
    The main RESTful endpoint for searching a single spectrum.
    Expected payload: {"spectrum": {...}, "library": "...", "method": "...", "metadata_filter": {...}, "search_params": {...}}
    The 'precursorMz' key in spectrum is optional.
    'search_params' is optional and can override server defaults.
    """
    data = request.get_json()
    
    if not data or 'spectrum' not in data or 'library' not in data:
        return jsonify({"error": "Invalid JSON payload. Missing 'spectrum' or 'library' keys."}), 400

    query_data = data.get('spectrum')
    library_name = data.get('library')
    search_method = data.get('method', 'matchms_cosine')
    metadata_filter = data.get('metadata_filter')

    log_usage(request.endpoint, request.method, library_name, 1, metadata_filter)

    # --- MODIFIED: Extract search parameters ---
    try:
        search_params = data.get('search_params', {})
        precursor_tol = float(search_params.get('precursor_mz_tolerance', PRECURSOR_MZ_TOLERANCE))
        fragment_tol = float(search_params.get('fragment_mz_tolerance', FRAGMENT_MZ_TOLERANCE))
        min_peaks = int(search_params.get('min_matched_peaks', MIN_MATCHED_PEAKS))
        top_n = int(search_params.get('combined_search_candidates', COMBINED_SEARCH_CANDIDATES))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid format for one or more search_params."}), 400
    # --- End Modification ---

    precursor_mz = query_data.get('precursorMz') # Will be None if not provided
    peaks = query_data.get('peaks')
    
    if peaks is None:
        return jsonify({"error": "Spectrum must have 'peaks' key"}), 400

    try:
        query_mz_array = np.array([p[0] for p in peaks], dtype=float)
        query_intensities_array = np.array([p[1] for p in peaks], dtype=float)
        
        precursor_float = None
        if precursor_mz is not None:
            precursor_float = float(precursor_mz)

        query_spectrum = Spectrum(mz=query_mz_array,
                                  intensities=query_intensities_array,
                                  metadata={'precursor_mz': precursor_float})
    except (ValueError, TypeError, IndexError) as e:
        return jsonify({"error": f"Invalid format for peaks or precursorMz: {e}"}), 400

    # --- MODIFIED: Pass params to perform_search ---
    result, status_code = perform_search(
        query_spectrum, library_name, search_method, metadata_filter,
        precursor_tol, fragment_tol, min_peaks, top_n
    )
    return jsonify(result), status_code

@app.route('/search/file', methods=['POST'])
def search_file_endpoint():
    """
    Endpoint to search spectra from an uploaded mzML file.
    Form data can include 'precursor_mz_tolerance', 'fragment_mz_tolerance', etc.
    to override server defaults.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not file.filename.lower().endswith('.mzml'):
        return jsonify({"error": "Invalid file type. Only mzML files are supported."}), 400

    library_name = request.form.get('library')
    search_method = request.form.get('method', 'matchms_cosine')
    
    # --- MODIFIED: Extract search parameters from form data ---
    try:
        precursor_tol = float(request.form.get('precursor_mz_tolerance', PRECURSOR_MZ_TOLERANCE))
        fragment_tol = float(request.form.get('fragment_mz_tolerance', FRAGMENT_MZ_TOLERANCE))
        min_peaks = int(request.form.get('min_matched_peaks', MIN_MATCHED_PEAKS))
        top_n = int(request.form.get('combined_search_candidates', COMBINED_SEARCH_CANDIDATES))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid format for one or more search_params in form data."}), 400
    # --- End Modification ---

    metadata_filter_str = request.form.get('metadata_filter')
    metadata_filter = None
    if metadata_filter_str:
        try:
            metadata_filter = json.loads(metadata_filter_str)
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid JSON format for metadata_filter."}), 400

    if not library_name:
        return jsonify({"error": "Missing 'library' in form data"}), 400

    try:
        # Load spectra from the mzML file
        mzml_spectra = list(matchms.importing.load_from_mzml(file.stream))
        
        log_usage(request.endpoint, request.method, library_name, len(mzml_spectra), metadata_filter)

        all_results = []
        for query_spectrum in mzml_spectra:
            # --- MODIFIED: Pass params to perform_search ---
            result, _ = perform_search(
                query_spectrum, library_name, search_method, metadata_filter,
                precursor_tol, fragment_tol, min_peaks, top_n
            )
            all_results.append(result)
            
        return jsonify(all_results)
    except Exception as e:
        app.logger.error(f"Error processing mzML file: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred while processing the mzML file: {e}"}), 500

@app.route('/search/batch', methods=['POST'])
def search_batch_endpoint():
    """
    Endpoint to search a batch (list) of spectra.
    Expected payload: {"spectra": [...], "library": "...", "search_params": {...}}
    'search_params' is optional.
    """
    data = request.get_json()
    
    if not data or 'spectra' not in data or 'library' not in data:
        return jsonify({"error": "Invalid JSON payload. Missing 'spectra' or 'library' keys."}), 400

    query_list = data.get('spectra')
    library_name = data.get('library')
    search_method = data.get('method', 'matchms_cosine')
    metadata_filter = data.get('metadata_filter')
    
    if not isinstance(query_list, list):
         return jsonify({"error": "'spectra' key must be a list."}), 400
    
    log_usage(request.endpoint, request.method, library_name, len(query_list), metadata_filter)

    # --- MODIFIED: Extract search parameters ---
    try:
        search_params = data.get('search_params', {})
        precursor_tol = float(search_params.get('precursor_mz_tolerance', PRECURSOR_MZ_TOLERANCE))
        fragment_tol = float(search_params.get('fragment_mz_tolerance', FRAGMENT_MZ_TOLERANCE))
        min_peaks = int(search_params.get('min_matched_peaks', MIN_MATCHED_PEAKS))
        top_n = int(search_params.get('combined_search_candidates', COMBINED_SEARCH_CANDIDATES))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid format for one or more search_params."}), 400
    # --- End Modification ---

    all_results = []
    for i, query_data in enumerate(query_list):
        if not isinstance(query_data, dict):
            all_results.append({"error": f"Item at index {i} is not a valid spectrum object."})
            continue

        precursor_mz = query_data.get('precursorMz') # Optional
        peaks = query_data.get('peaks')
        
        if peaks is None:
            all_results.append({"error": f"Spectrum at index {i} is missing 'peaks'."})
            continue

        try:
            query_mz_array = np.array([p[0] for p in peaks], dtype=float)
            query_intensities_array = np.array([p[1] for p in peaks], dtype=float)
            
            precursor_float = None
            if precursor_mz is not None:
                precursor_float = float(precursor_mz)
            
            query_spectrum = Spectrum(mz=query_mz_array,
                                      intensities=query_intensities_array,
                                      metadata={'precursor_mz': precursor_float, 'name': f'batch_spectrum_{i}'})
        except (ValueError, TypeError, IndexError) as e:
            all_results.append({"error": f"Invalid format for spectrum at index {i}: {e}"})
            continue
        
        # --- MODIFIED: Pass params to perform_search ---
        result, _ = perform_search(
            query_spectrum, library_name, search_method, metadata_filter,
            precursor_tol, fragment_tol, min_peaks, top_n
        )
        all_results.append(result)
        
    return jsonify(all_results)

# --- Basic Web GUI ---
@app.route('/')
def ui():
    # --- MODIFIED: Define HTML_TEMPLATE inside the request context ---
    # This allows us to inject the default config values using an f-string
    HTML_TEMPLATE = f"""
    <!doctype html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <title>MS2 Search</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 900px; margin: 40px auto; background-color: #fdfdfd; color: #333; }}
        h2, h3 {{ color: #005a9c; border-bottom: 2px solid #eee; padding-bottom: 5px; }}
        input, select, button, textarea {{ 
            margin: 6px 0; 
            padding: 8px; 
            width: 100%; 
            box-sizing: border-box; /* Important */
            border: 1px solid #ccc; 
            border-radius: 4px;
            font-family: inherit;
            font-size: 1rem;
        }}
        button {{ 
            background-color: #0078d4; 
            color: white; 
            font-weight: bold; 
            border: none; 
            cursor: pointer;
            transition: background-color 0.2s;
        }}
        button:hover {{ background-color: #005a9c; }}
        textarea {{ min-height: 150px; font-family: "Courier New", Courier, monospace; }}
        hr {{ border: 0; border-top: 1px solid #eee; margin: 25px 0; }}
        .result {{ 
            white-space: pre-wrap; 
            background: #f4f4f4; 
            padding: 15px; 
            border-radius: 4px; 
            border: 1px solid #ddd;
            max-height: 600px;
            overflow-y: auto;
            font-family: "Courier New", Courier, monospace;
        }}
        .upload-section {{ margin-top: 15px; }}
        label {{ font-weight: bold; display: block; margin-top: 10px; }}
        
        /* --- NEW CSS --- */
        .param-grid {{ 
            display: grid; 
            grid-template-columns: 1fr 1fr; 
            gap: 15px; 
            background: #f9f9f9;
            border: 1px solid #eee;
            padding: 15px;
            border-radius: 4px;
        }}
        .param-grid label {{ margin-top: 0; font-size: 0.9em; }}
        .param-grid input {{ margin-top: 2px; padding: 6px; }}
        #uploadStatus {{ font-size: 0.9em; color: #333; margin-top: 5px; }}
        /* --- End NEW CSS --- */
    </style>
    </head>
    <body>
    <h2>MS2 Spectral Search Portal</h2>

    <label for="library">Library:</label>
    <select id="library"></select>

    <label for="method">Search method:</label>
    <select id="method">
        <option value="matchms_cosine">MatchMS Cosine</option>
        <option value="flash_entropy">Flash Entropy</option>
        <option value="combined">Combined (Entropy Filter + Cosine)</option>
    </select>
    
    <hr>
    <h3>Search Parameters (Advanced)</h3>
    <div class="param-grid">
        <div>
            <label for="precursor_tol">Precursor m/z Tolerance:</label>
            <input type="number" step="0.01" id="precursor_tol" value="{PRECURSOR_MZ_TOLERANCE}">
        </div>
        <div>
            <label for="fragment_tol">Fragment m/z Tolerance:</label>
            <input type="number" step="0.01" id="fragment_tol" value="{FRAGMENT_MZ_TOLERANCE}">
        </div>
        <div>
            <label for="min_peaks">Min. Matched Peaks:</label>
            <input type="number" step="1" id="min_peaks" value="{MIN_MATCHED_PEAKS}">
        </div>
        <div>
            <label for="top_n">Combined Search Candidates (Top N):</label>
            <input type="number" step="1" id="top_n" value="{COMBINED_SEARCH_CANDIDATES}">
        </div>
    </div>
    <hr>
    <h3>Option 1: Upload mzML File</h3>
    <div class="upload-section">
        <input type="file" id="fileInput" accept=".mzML">
        <button id="uploadBtn">Upload & Search File</button>
    </div>

    <hr>
    <h3>Option 2: Paste mz,intensity pairs (for one spectrum)</h3>
    <label for="pairs">Peak List (mz intensity or mz,intensity)</label>
    <textarea id="pairs" placeholder="mz intensity\n100.1 200\n101.5 150\n..."></textarea>
    <label for="precursor">Precursor m/z</label>
    <input id="precursor" type="number" placeholder="Precursor m/z (optional, e.g., for EI-MS)">
    <button id="submitBtn">Run Single Search</button>

    <hr>
    <h3>Option 3: Upload New Library</h3>
    <div class="upload-section">
        <label for="libraryFileInput">Select .msp library file:</label>
        <input type="file" id="libraryFileInput" accept=".msp">
        <button id="uploadLibraryBtn">Upload Library</button>
        <div id="uploadStatus"></div>
    </div>
    <hr>
    <h3>Results</h3>
    <div id="output" class="result">[Results will appear here]</div>

    <script>
        // --- NEW: Helper function to get search params ---
        function getSearchParams() {{
            return {{
                "precursor_mz_tolerance": parseFloat(document.getElementById('precursor_tol').value),
                "fragment_mz_tolerance": parseFloat(document.getElementById('fragment_tol').value),
                "min_matched_peaks": parseInt(document.getElementById('min_peaks').value),
                "combined_search_candidates": parseInt(document.getElementById('top_n').value)
            }};
        }}

        async function loadLibs(){{
            try {{
                const r = await fetch('/libraries');
                const j = await r.json();
                const s = document.getElementById('library');
                s.innerHTML = ''; // Clear existing
                if (j.available_libraries && j.available_libraries.length > 0) {{
                    j.available_libraries.forEach(l => {{
                        let o = document.createElement('option');
                        o.value = l;
                        o.textContent = l;
                        s.appendChild(o);
                    }});
                }} else if (j.warning) {{
                    let o = document.createElement('option');
                    o.textContent = j.warning;
                    o.disabled = true;
                    s.appendChild(o);
                }}
            }} catch (e) {{
                document.getElementById('output').textContent = "Error loading libraries: " + e.message;
            }}
        }}

        function showOutput(data) {{
            document.getElementById('output').textContent = JSON.stringify(data, null, 2);
        }}
        
        function showLoading() {{
            document.getElementById('output').textContent = "Searching...";
        }}

        async function runTextSearch(){{
            const txt = document.getElementById('pairs').value.trim();
            if(!txt) {{
                alert('Please enter a peak list.');
                return;
            }}
            
            const precursor_val = document.getElementById('precursor').value;
            const precursor = precursor_val ? parseFloat(precursor_val) : null;
            
            const peaks = txt.split(/\\n+/)
                             .map(l => l.trim().split(/[,\\s]+/))
                             .map(parts => parts.map(parseFloat))
                             .filter(a => a.length === 2 && !isNaN(a[0]) && !isNaN(a[1]));
            
            if (peaks.length === 0) {{
                alert('No valid (mz, intensity) pairs found. Please use format: mz intensity');
                return;
            }}

            // --- MODIFIED: Add search_params to payload ---
            const payload = {{
                spectrum: {{ precursorMz: precursor, peaks: peaks }},
                library: document.getElementById('library').value,
                method: document.getElementById('method').value,
                search_params: getSearchParams() 
            }};
            
            showLoading();
            try {{
                const res = await fetch('/search', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify(payload)
                }});
                showOutput(await res.json());
            }} catch (e) {{
                showOutput({{ error: e.message }});
            }}
        }}

        async function uploadFile(){{
            const f = document.getElementById('fileInput').files[0];
            if(!f) {{
                alert('Please select a .mzML file to upload.');
                return;
            }}

            const fd = new FormData();
            fd.append('file', f);
            fd.append('library', document.getElementById('library').value);
            fd.append('method', document.getElementById('method').value);
            
            // --- MODIFIED: Add search params to FormData ---
            const params = getSearchParams();
            fd.append('precursor_mz_tolerance', params.precursor_mz_tolerance);
            fd.append('fragment_mz_tolerance', params.fragment_mz_tolerance);
            fd.append('min_matched_peaks', params.min_matched_peaks);
            fd.append('combined_search_candidates', params.combined_search_candidates);
            
            showLoading();
            try {{
                const res = await fetch('/search/file', {{
                    method: 'POST',
                    body: fd
                }});
                showOutput(await res.json());
            }} catch (e) {{
                showOutput({{ error: e.message }});
            }}
        }}

        async function uploadLibrary() {{
            const lib = document.getElementById('libraryFileInput').files[0];
            if (!lib) {{
                alert('Please select a .msp file to upload.');
                return;
            }}
            const fd = new FormData();
            fd.append('file', f);

            const statusDiv = document.getElementById('uploadStatus');
            statusDiv.textContent = 'Uploading...';

            try {{
                const res = await fetch('/libraries/upload', {{
                    method: 'POST',
                    body: fd
                }});
                const result = await res.json(); // 'result' holds the JSON
                
                if (res.ok) {{
                    // --- FIX: Changed res.message to result.message ---
                    statusDiv.textContent = `Status: ${{result.message}}`;
                    if (result.status === 'success') {{
                        loadLibs(); // Reload the library list
                    }}
                }} else {{
                    // --- FIX: Changed to check result.error and result.message ---
                    // This handles all error messages from the server
                    statusDiv.textContent = `Error: ${{result.error || result.message}}`;
                }}
            }} catch (e) {{
                statusDiv.textContent = `Error: ${{e.message}}`;
            }}
        }}
        document.getElementById('submitBtn').onclick = runTextSearch;
        document.getElementById('uploadBtn').onclick = uploadFile;
        document.getElementById('uploadLibraryBtn').onclick = uploadLibrary; 
        loadLibs();
    </script>
    </body></html>
    """
    return render_template_string(HTML_TEMPLATE)


if __name__ == '__main__':
    # Initial check for folders
    if not os.path.exists(LIBRARY_FOLDER):
        os.makedirs(LIBRARY_FOLDER)
        print(f"Created directory: {LIBRARY_FOLDER}")
    if not os.path.exists(LOGS_FOLDER):
        os.makedirs(LOGS_FOLDER)
        print(f"Created directory: {LOGS_FOLDER}")

    print(f"Checking for libraries in '{LIBRARY_FOLDER}'...")
    libs = get_available_libraries()
    if libs:
        print(f"Found libraries: {', '.join(libs)}")
    else:
        print("No libraries found. Upload via the GUI or add .msp files to the 'libraries' folder.")
    
    print("Server starting on http://0.0.0.0:5000")
    # For production, use a proper WSGI server like Gunicorn:
    # gunicorn --workers 4 --bind 0.0.0.0:5000 server:app
    app.run(host='0.0.0.0', port=5000, debug=True)