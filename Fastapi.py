from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Note(BaseModel):
    title: str
    content: str
    category: str = "general"


# Store notes here
notes = {}


@app.post("/notes/{note_id}")
def create_notes(note_id: int, note: Note):
    notes[note_id] = note

    return {
        "message": "Note Created",
        "note": {
            "id": note_id,
            "title": note.title,
            "content": note.content,
            "category": note.category
        }
    }


@app.get("/notes/{note_id}")
def get_notes(note_id: int):
    note = notes.get(note_id)

    return {
        "id": note_id,
        "title": note.title,
        "content": note.content,
        "category": note.category
    }