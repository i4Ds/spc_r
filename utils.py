import concurrent.futures
import hashlib
import json
import os
import re

import faiss  # type: ignore
import numpy as np
import PyPDF2
from openai import OpenAI  # type: ignore
from tqdm import tqdm

client = OpenAI()

# --- CONSTANTS ---
PROMPT_V3 = """ 
Du bist ein Schweizerdeutsch Audio-zu-Text-Korrekturmeister, welcher die Transkription Teil-für-Teil korrigiert. Zur Unterstützung erhältst du:
1. Einen oder mehrere relevante Ausschnitte aus einer manuell erstellten Zusammenfassung einer politischen Sitzung. Bedenke, dass diese Zusammenfassungen mehr Informationen beinhalten können als die Transkription, da die Ausschnitte aus späteren oder früheren Zeitpunkten der Sitzung stammen können.
2. Einen Teil der Transkription der politischen Sitzung.

Überlege zuerst, ob irgendwelche Ausschnitte relevant für die Transkription sind, auf denen du die Korrektur basieren kannst.
Relevant bedeutet, dass die Ausschnitte aus der Zusammenfassung den Teil der Transkription beinhalten, den du jetzt korrigieren sollst. Falls nicht, antworte nur mit "Keine relevanten Ausschnitte".
Korrigiere nur das, was du anhand des Ausschnitts mit Sicherheit korrigieren kannst.
Falls die Transkription nicht auf Deutsch ist, antworte nur mit "Keine Deutsche Transkription".
Falls keine Korrektur nötig ist, da alle Regeln der Korrektur (siehe die Regeln 1-5 unten) schon erfüllt sind, antworte nur mit "Keine Korrektur notwendig."

Folge folgenden Regeln für die Korrektur:

1. Überprüfe, ob in der Transkription anhand der Zusammenfassung Fehler bei Nomen, Zahlen oder Abkürzungen vorkommen. Diese Art der Korrektur ist am wichtigsten. Bevor du eine Zahl als Fehler markierst, überprüfe, ob die Bedeutung der Zahl identisch ist (z.B. "8:15" und "Viertel nach acht" oder "Jahr 26" und "Jahr 2026").
2. Überprüfe, ob in der Transkription folgende schweizerdeutsche Besonderheiten vorkommen und passe diese an:
   - übermässige Verwendung von "ihr" anstatt "Sie".
   - übermässige Verwendung von "euch" anstatt "Sie".
   - "Ier", welches transkribiert wird als "er", muss manchmal als "ihr" korrigiert werden, falls es im Satz Sinn ergibt.
   - "vo dr", welches transkribiert wird als "vor der", muss als "von der" korrigiert werden, falls es im Satz Sinn ergibt.
   - "vo", welches transkribiert wird als "vor", muss manchmal als "von" korrigiert werden, falls es im Satz Sinn ergibt.
   - "Mier", welches transkribiert wird als "Mir", muss manchmal als "wir" korrigiert werden, falls es im Satz Sinn ergibt.
   - Im Schweizerdeutschen lässt man oft das "ge" weg bei Verben, also "usglaugt" wird transkribiert als "ausglaugt", muss als "ausgeglaut" korrigiert werden. Oder "usdüütscht" wird transkribiert als "ausdeutscht", muss als "ausgedeutscht" korrigiert werden.
3. Korrigiere ausschliesslich diese Fehler und lasse den übrigen Wortlaut und die Intention des Audios unverändert, um eine konsistente Grundbasis für das Training eines Speech-to-Text-Modells zu gewährleisten.
4. Falls eine Korrektur nötig und möglich ist, antworte mit der ganzen, korrigierten Transkription.
5. Gewisse Wörter können gleich klingen im Schweizerdeutschen, aber anders geschrieben werden und eine andere Bedeutung haben. In diesem Fall überlege dir, welche Bedeutung im Kontext am sinnvollsten ist, und korrigiere das Wort entsprechend. 
   Beispiel: "Tour" wird transkribiert als "Tour". In der Transkription würde aber "Dauer" am sinnvollsten sein, was auch als "Tour" ausgesprochen wird im Schweizerdeutschen, also korrigiere "Tour" zu "Dauer".

Halte dich sehr genau an die Transkription! Füge keine neuen Informationen hinzu, welche du aus den relevanten Ausschnitten entnehmen könntest, und entferne keine Informationen, die in der Transkription enthalten sind. 
Ersetze NIE den Originaltext durch eine Zusammenfassung oder eine andere Formulierung. Ersetze KEINE Wörter durch Synonyme, auch wenn diese in den relevanten Ausschnitten vorkommen.
Verändere NIE die Wortfolge oder die Satzstruktur. Entferne KEINE Wörter, auch wenn diese in den relevanten Ausschnitten nicht vorkommen.
Bedenke immer, dass diese Daten für das Training eines Speech-to-Text-Modells verwendet werden, somit MUSS die Korrektur so nah wie möglich am originalen Wortlaut aus dem Audio bleiben.

"""

PROMPT_RATING_2 = """
Du erhältst zwei Texte:

1. Einen oder mehrere relevante Ausschnitte aus einer manuell erstellten Zusammenfassung einer politischen Sitzung. Bedenke, dass diese Zusammenfassungen irrelevante informationen beinhalten, da die Ausschnitte aus späteren oder früheren Zeitpunkten der Sitzung stammen können.
2. Ein Teil der Transkription der politischen Sitzung.

Bewerte ausschliesslich, ob der Teil der Transkription in Bezug auf Namen, Nomen, Zahlen und Abkürzungen mit dem relevanten Ausschnitten semantisch identisch ist. Verwende dazu folgende Kategorien von 0 bis 3:

- 3: Alle Namen, Nomen, Zahlen, Abkürzungen in der Transkription sind korrekt und es hat keinen Fehler.
- 2: Es gibt einen sonstigen kleinen Fehler (z. B. in einer Konjugation oder ein falscher Artikel), aber alle Namen, Nomen, Zahlen und Abkürzungen sind korrekt,
- 1: Ein oder mehrere Namen, Nomen, Zahlen oder Abkürzungen sind falsch geschrieben.
- 0: Es sind keine relevanten Ausschnitte vorhanden und / oder die relevanten Ausschnitte sind in einer anderen Sprache als Deutsch, somit ist weder eine Korrektur, noch eine Bewertung möglich.

Regeln: 
- Gross- und Kleinschreibung ist kein Fehler.
- Unterschiedliche Schreibweisen von Zahlen (z.B. "8:15" vs. "Viertel nach Acht" oder "im Jahr Sechsundzwanzig" vs. "im 26") zählen nicht als Fehler.
- Ausgeschriebene Zahlen vs. Ziffern sind ebenfalls kein Fehler (z.B. "vier" vs. "4")
- Ausgeschriebene Abkürzungen (z.B. "z.B." vs. "zum Beispiel") sind ebenfalls kein Fehler.
- Leicht anders zusammengesetzte Wörter, wie z.B. "Links-Grün" vs "Linksgrün" oder "MittelLinks" vs "Mitte-Links", sind keine Fehler.

Um zu bewerten, fange am besten bei 0 an und falls dieser Punkt nicht zutrifft, gehe zu 1 und so weiter, bis die Regeln eines Punktes zutreffen.
Antworte ausschliesslich mit der entsprechenden Zahl! Ignoriere Fehler in den Ausschnitten, bewerte AUSCHLIESSLICH NUR die Transkription!
"""

NO_CORR_SIGNALS = {
    "keinekorrekturnotwendig",
    "keinerelevantenausschnitte",
    "keinedeutschetranskription",
}


# --- PDF AND TEXT PROCESSING ---
def merge_hyphenated_words(text):
    """Merge words split by hyphen and newline."""
    pattern = r"(\w+)-\n(\w+)"
    return re.sub(pattern, r"\1\2", text)


def extract_text_from_pdf(pdf_path):
    pages = []
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text().strip()
            # Correct hyphenation breaks over line breaks:
            page_text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", page_text)
            page_text = merge_hyphenated_words(page_text)
            page_text = re.sub(r"\s+", " ", page_text)
            page_text = re.sub(r"\n+", "\n", page_text)
            page_text = re.sub(r"\.{2,}", ".", page_text)
            pages.append(page_text)
    return "\n".join(pages)


def save_text_to_file(text, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(text)


def chunk_text(text, chunk_size=600, overlap=450):
    """Return a list of overlapping chunks (by character count)."""
    text = text.replace("\n", " ")  # Remove newlines
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# --- EMBEDDING & FAISS INDEX ---
def get_cache_id(model_name, chunks):
    """Generate a SHA-256 hash based on the model name and text chunks."""
    sha = hashlib.sha256()
    sha.update(model_name.encode("utf-8"))
    for chunk in chunks:
        sha.update(chunk.encode("utf-8"))
    return sha.hexdigest()


def create_vector_store(chunks, base_folder="embeddings"):
    """Create a FAISS index of embeddings for the text chunks.
    Cached embeddings are stored in 'base_folder' to avoid repeated API calls.
    """
    model_name = "text-embedding-3-large"
    cache_id = get_cache_id(model_name, chunks)
    os.makedirs(base_folder, exist_ok=True)
    cache_file = os.path.join(base_folder, f"embeddings_{cache_id}.npy")

    if os.path.exists(cache_file):
        print("Loading cached embeddings")
        embeddings = np.load(cache_file)
    else:
        print("Generating embeddings (this may take time)...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            embeddings = list(
                tqdm(
                    executor.map(
                        lambda chunk: get_embedding(chunk, model_name), chunks
                    ),
                    total=len(chunks),
                )
            )
        embeddings = np.array(embeddings, dtype="float32")
        np.save(cache_file, embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return model_name, index


def get_embedding(chunk, model_name):
    response = client.embeddings.create(input=chunk, model=model_name)
    return response.data[0].embedding


def retrieve_relevant_chunks(query, model_name, index, chunks, top_k=1):
    """Return the top_k chunks that are most relevant to the query."""
    response = client.embeddings.create(input=[query], model=model_name)
    query_embedding = response.data[0].embedding
    query_vector = np.array([query_embedding], dtype="float32")
    _, indices = index.search(query_vector, top_k)
    return [chunks[i] for i in indices[0]]


# --- PROMPT BUILDING ---
def build_llm_prompt(segment, relevant_chunks):
    chunks_part = "Relevante Ausschnitte aus der Zusammenfassung:"
    for chunk in relevant_chunks:
        chunks_part += f"\n\nAusschnitt: {chunk}"
    return chunks_part + f"\n\nTranskription zum Korrigieren: {segment}"


def build_rating_llm_prompt_2(segment, relevant_chunks):
    chunks_part = "Relevante Ausschnitte aus der Zusammenfassung:"
    for chunk in relevant_chunks:
        chunks_part += f"\n\nAusschnitt: {chunk}"
    return f"{chunks_part}\n\nTranskription: {segment}"


def normalize_text(text):
    return re.sub(r"[^a-zäöüß]", "", text.lower())


def get_judgement_text(seg_data):
    original_text = seg_data["text"].strip()
    corrected = seg_data["corrected_text"]
    if normalize_text(corrected) in NO_CORR_SIGNALS:
        return original_text
    return corrected


def prepare_messages(seg_data, step, embedding_model, faiss_index, text_chunks):
    """
    Build the messages for either a correction or a judgement step.
    Returns the list of messages and the retrieved relevant chunks.
    """
    original_text = seg_data["text"].strip()
    relevant_chunks = retrieve_relevant_chunks(
        original_text, embedding_model, faiss_index, text_chunks, top_k=1
    )
    if step == "correction":
        system_msg = PROMPT_V3.strip()
        user_msg = build_llm_prompt(original_text, relevant_chunks)
    elif step == "judgement":
        judged_text = get_judgement_text(seg_data)
        system_msg = PROMPT_RATING_2.strip()
        user_msg = build_rating_llm_prompt_2(judged_text, relevant_chunks)
    else:
        raise ValueError("Unknown step: " + step)
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    return messages, relevant_chunks

