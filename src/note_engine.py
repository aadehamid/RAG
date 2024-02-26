
import os
from pathlib import Path

from llama_index.core.tools import FunctionTool

# note_file = os.path.join("..", "agent_data", "notes.txt")
note_file = Path("/Users/hamidadesokan/Dropbox/2_Skill_Development/DLML/genai_applications/" \
                 "embeddings/RAG/agent/agent_data/notes.txt")
# note_file = Path.joinpath(Path.cwd(), "RAG", "agent", 'agent_data', "notes.txt")


def save_note(note):
    if not os.path.exists(note_file):
        open(note_file, "w")

    with open(note_file, "a") as f:
        f.writelines([note + "\n"])

    return "note saved"


note_engine = FunctionTool.from_defaults(
    fn=save_note,
    name="note_saver",
    description="this tool can save a text based note to a file for the user",
)