# MS2 Spectral Search REST API using Flask, MatchMS, and MSEntropy
#
# This script creates a web server that provides a RESTful interface
# for searching MS2 spectra against MSP libraries. It now includes
# Flash Entropy Search and API usage tracking for grant reporting.
#
# To run this application:
# 1. Make sure you have Python installed.
# 2. Install the required libraries:
#    pip install Flask matchms ms_entropy
# 3. Create a folder named `libraries` and `logs` in the same directory as this script.
# 4. Place your MSP spectral library files (e.g., `my_lib.msp`) inside the `libraries` folder.
# 5. Run this script: python your_script_name.py
# 6. The server will start on http://127.0.0.1:5000.

from flask import Flask, request, jsonify
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

# --- Configuration ---
# The folder where your .msp library files are stored.
LIBRARY_FOLDER = './libraries'
# The folder for log files.
LOGS_FOLDER = './logs'
# Log file name for API usage.
USAGE_LOG_FILE = os.path.join(LOGS_FOLDER, 'api_usage.log')
# Precursor m/z tolerance for finding candidate spectra.
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
    """
    best_match = None
    best_score = 0.0
    
    query_precursor_mz = query_spectrum.get('precursor_mz')
    if not query_precursor_mz:
        return None, 0.0

    # 1. Filter library by precursor m/z to get candidate spectra
    candidate_spectra = [spec for spec in library_spectra if abs(spec.get('precursor_mz', 0) - query_precursor_mz) <= precursor_mz_tolerance]
    
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
    """
    query_metadata = query_spectrum.metadata.copy()
    library_metadata = [spec.metadata for spec in library_spectra]

    # Convert matchms spectrum objects to the format expected by ms-entropy
    query_data = {'precursor_mz': query_metadata.get('precursor_mz'), 'peaks': list(zip(query_spectrum.mz, query_spectrum.intensities))}
    library_data = [{'precursor_mz': lib.get('precursor_mz'), 'peaks': list(zip(lib.mz, lib.intensities))} for lib in library_spectra]
    
    # Perform the search
    search_results = ms_entropy.flash_entropy_search(
        query_data, 
        library_data, 
        precursor_mz_tolerance=precursor_mz_tolerance
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
    top_n_candidates: int
) -> tuple[Union[Spectrum, None], float]:
    """
    Performs a two-stage search:
    1. Initial fast filter with Flash Entropy to get top N candidates.
    2. Refine the search with CosineGreedy on the candidates.
    """
    # Stage 1: Fast filtering with Flash Entropy to get top N candidates
    query_data = {'precursor_mz': query_spectrum.get('precursor_mz'), 'peaks': list(zip(query_spectrum.mz, query_spectrum.intensities))}
    library_data = [{'precursor_mz': lib.get('precursor_mz'), 'peaks': list(zip(lib.mz, lib.intensities))} for lib in library_spectra]
    
    # Get top N search results from Flash Entropy
    entropy_results = ms_entropy.flash_entropy_search(
        query_data,
        library_data,
        precursor_mz_tolerance=precursor_mz_tolerance
    )

    if not entropy_results:
        return None, 0.0

    # Extract the top N candidates and their indices
    candidate_indices = [idx for idx, _ in entropy_results[:top_n_candidates]]
    candidate_spectra = [library_spectra[idx] for idx in candidate_indices]

    if not candidate_spectra:
        return None, 0.0

    # Stage 2: High-fidelity refinement with Matchms CosineGreedy
    cosine_greedy = CosineGreedy(tolerance=FRAGMENT_MZ_TOLERANCE)
    scores = cosine_greedy.matrix([query_spectrum], candidate_spectra)
    
    best_match = None
    best_score = 0.0

    if scores.shape[1] > 0:
        best_match_idx = np.argmax(scores[0, :])
        score_tuple = scores[0, best_match_idx]
        
        # Check if the match meets the minimum peak requirement
        if score_tuple[1] >= MIN_MATCHED_PEAKS:
            best_score = score_tuple[0]
            best_match = candidate_spectra[best_match_idx]
    
    return best_match, best_score

# --- General Search Function ---
def perform_search(
    query_spectrum: Spectrum, 
    library_name: str,
    search_method: str = 'matchms_cosine',
    metadata_filter: Optional[Dict[str, Any]] = None
) -> tuple[Dict[str, Any], int]:
    """
    Performs a search on a single spectrum against a given library, with optional metadata filtering.
    """
    try:
        if library_name not in get_available_libraries():
            return {'error': f"Library '{library_name}' not found. Available libraries: {get_available_libraries()}"}, 404

        lib_spectra = load_and_process_library(library_name)

        # Apply metadata filter if provided
        if metadata_filter:
            filtered_lib_spectra = [
                s for s in lib_spectra if all(
                    s.get(k) == v for k, v in metadata_filter.items()
                )
            ]
            if not filtered_lib_spectra:
                return {'status': 'no_match_found', 'reason': 'No spectra match the provided metadata filter.'}, 404
        else:
            filtered_lib_spectra = lib_spectra

        processed_query_spectrum = process_spectrum(query_spectrum)
        if processed_query_spectrum is None:
            return {'status': 'no_match_found', 'reason': 'Query spectrum is empty after processing.'}, 404

        if search_method == 'matchms_cosine':
            best_match, best_score = search_spectrum_with_matchms(
                processed_query_spectrum, 
                filtered_lib_spectra,
                PRECURSOR_MZ_TOLERANCE,
                FRAGMENT_MZ_TOLERANCE,
                MIN_MATCHED_PEAKS
            )
        elif search_method == 'flash_entropy':
             best_match, best_score = search_spectrum_with_entropy(
                processed_query_spectrum,
                filtered_lib_spectra,
                PRECURSOR_MZ_TOLERANCE
            )
        elif search_method == 'combined':
            best_match, best_score = search_spectrum_with_combined(
                processed_query_spectrum,
                filtered_lib_spectra,
                PRECURSOR_MZ_TOLERANCE,
                COMBINED_SEARCH_CANDIDATES
            )
        else:
            return {'error': f"Invalid search method '{search_method}'. Choose 'matchms_cosine', 'flash_entropy', or 'combined'."}, 400
            
        if best_match:
            result = {
                'status': 'match_found',
                'library': library_name,
                'search_method': search_method,
                'score': round(best_score, 4),
                'annotation': best_match.metadata,
            }
            return result, 200
        else:
            return {'status': 'no_matches', 'search_method': search_method}, 200
    
    except FileNotFoundError as e:
        return {'error': str(e)}, 404
    except Exception as e:
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


@app.route('/search', methods=['POST'])
def search_endpoint():
    """
    The main RESTful endpoint for searching a single spectrum.
    Expected payload: {"spectrum": {...}, "library": "...", "method": "...", "metadata_filter": {...}}
    The metadata_filter is an optional key-value dictionary to pre-filter library spectra.
    """
    data = request.get_json()
    
    if not data or 'spectrum' not in data or 'library' not in data:
        return jsonify({"error": "Invalid JSON payload. Missing 'spectrum' or 'library' keys."}), 400

    query_data = data.get('spectrum')
    library_name = data.get('library')
    search_method = data.get('method', 'matchms_cosine') # Default to matchms
    metadata_filter = data.get('metadata_filter')

    log_usage(request.endpoint, request.method, library_name, 1, metadata_filter)

    precursor_mz = query_data.get('precursorMz')
    peaks = query_data.get('peaks')
    
    if precursor_mz is None or peaks is None:
        return jsonify({"error": "Spectrum must have 'precursorMz' and 'peaks' keys"}), 400

    try:
        query_mz_array = np.array([p[0] for p in peaks], dtype=float)
        query_intensities_array = np.array([p[1] for p in peaks], dtype=float)
        
        query_spectrum = Spectrum(mz=query_mz_array,
                                  intensities=query_intensities_array,
                                  metadata={'precursor_mz': float(precursor_mz)})
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid format for peaks or precursorMz: {e}"}), 400

    result, status_code = perform_search(query_spectrum, library_name, search_method, metadata_filter)
    return jsonify(result), status_code

@app.route('/search/mzml', methods=['POST'])
def search_mzml_endpoint():
    """
    Endpoint to search spectra from an uploaded mzML file.
    The mzML file should be sent in the request body as form data.
    Expected form data: {"file": <mzml_file>, "library": "...", "method": "...", "metadata_filter": "{...}"}
    The metadata_filter is a stringified JSON object.
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
    metadata_filter_str = request.form.get('metadata_filter')
    
    if metadata_filter_str:
        try:
            metadata_filter = json.loads(metadata_filter_str)
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid JSON format for metadata_filter."}), 400
    else:
        metadata_filter = None

    if not library_name:
        return jsonify({"error": "Missing 'library' in form data"}), 400

    try:
        # Load spectra from the mzML file
        mzml_spectra = list(matchms.importing.load_from_mzml(file))
        
        log_usage(request.endpoint, request.method, library_name, len(mzml_spectra), metadata_filter)

        all_results = []
        for query_spectrum in mzml_spectra:
            result, _ = perform_search(query_spectrum, library_name, search_method, metadata_filter)
            all_results.append(result)
            
        return jsonify(all_results)
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred while processing the mzML file: {e}"}), 500

@app.route('/search/batch', methods=['POST'])
def search_batch_endpoint():
    """
    Endpoint to search a batch (list) of spectra.
    Expected payload: {"spectra": [{"precursorMz":...,"peaks":...}, ...], "library": "...", "method": "...", "metadata_filter": {...}}
    """
    data = request.get_json()
    
    if not data or 'spectra' not in data or 'library' not in data:
        return jsonify({"error": "Invalid JSON payload. Missing 'spectra' or 'library' keys."}), 400

    query_list = data.get('spectra')
    library_name = data.get('library')
    search_method = data.get('method', 'matchms_cosine')
    metadata_filter = data.get('metadata_filter')
    
    log_usage(request.endpoint, request.method, library_name, len(query_list), metadata_filter)

    all_results = []
    for query_data in query_list:
        precursor_mz = query_data.get('precursorMz')
        peaks = query_data.get('peaks')
        
        if precursor_mz is None or peaks is None:
            all_results.append({"error": "A spectrum in the batch is missing 'precursorMz' or 'peaks'."})
            continue

        try:
            query_mz_array = np.array([p[0] for p in peaks], dtype=float)
            query_intensities_array = np.array([p[1] for p in peaks], dtype=float)
            
            query_spectrum = Spectrum(mz=query_mz_array,
                                      intensities=query_intensities_array,
                                      metadata={'precursor_mz': float(precursor_mz)})
        except (ValueError, TypeError) as e:
            all_results.append({"error": f"Invalid format for a spectrum in the batch: {e}"})
            continue
        
        result, _ = perform_search(query_spectrum, library_name, search_method, metadata_filter)
        all_results.append(result)
        
    return jsonify(all_results)

if __name__ == '__main__':
    # Initial check for the library folder
    print(f"Checking for libraries in '{LIBRARY_FOLDER}'...")
    get_available_libraries()
    print("Server starting...")
    # For production, you should use a proper WSGI server like Gunicorn or uWSGI.
    # Example: gunicorn --workers 4 --bind 0.0.0.0:5000 app:app
    app.run(host='0.0.0.0', port=5000, debug=True)
