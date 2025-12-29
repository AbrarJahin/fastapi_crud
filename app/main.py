from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from app.database import SessionLocal
from app.schemas import ItemCreate, ItemUpdate, ItemOut, PaginatedItems
from app.models.db.item import Item

import logging

logger = logging.getLogger(__name__)

app = FastAPI(
    title="FastAPI SQLite CRUD",
    version="1.0.0",
    description="FastAPI + SQLAlchemy + SQLite + Alembic (Conda-managed)",
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/health")
def health():
    logger.info("Health check called")
    return {
        "status": "ok",
        "time": datetime.utcnow()
    }


# 1) CREATE
@app.post("/items", response_model=ItemOut, status_code=status.HTTP_201_CREATED)
def create_item(payload: ItemCreate, db: Session = Depends(get_db)):
    item = Item(name=payload.name, description=payload.description)
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


# 2) DETAIL BY ID
@app.get("/items/{item_id}", response_model=ItemOut)
def get_item(item_id: int, db: Session = Depends(get_db)):
    item = db.get(Item, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item


# 3) GET ALL with pagination (skip + limit) and total count
@app.get("/items", response_model=PaginatedItems)
def list_items(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
):
    total = db.scalar(select(func.count()).select_from(Item)) or 0
    items = db.scalars(select(Item).offset(skip).limit(limit)).all()
    return {"total": total, "skip": skip, "limit": limit, "items": items}


# 4) UPDATE (partial update)
@app.put("/items/{item_id}", response_model=ItemOut)
def update_item(item_id: int, payload: ItemUpdate, db: Session = Depends(get_db)):
    item = db.get(Item, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    if payload.name is not None:
        item.name = payload.name
    if payload.description is not None:
        item.description = payload.description

    db.add(item)
    db.commit()
    db.refresh(item)
    return item


# 5) DELETE
@app.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_item(item_id: int, db: Session = Depends(get_db)):
    item = db.get(Item, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    db.delete(item)
    db.commit()
    return None
