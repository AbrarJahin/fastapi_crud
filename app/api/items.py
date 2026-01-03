from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from app.database import SessionLocal
from app.models.view import ItemCreate, ItemUpdate, ItemOut, PaginatedItems
from app.models.db.item import Item

import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/items", tags=["Items"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 1) CREATE
@router.post("", response_model=ItemOut, status_code=status.HTTP_201_CREATED)
def create_item(payload: ItemCreate, db: Session = Depends(get_db)):
    logger.info("Create item called")
    item = Item(name=payload.name, description=payload.description)
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


# 2) DETAIL BY ID
@router.get("/{item_id}", response_model=ItemOut)
def get_item(item_id: int, db: Session = Depends(get_db)):
    logger.info("Get item called: %s", item_id)
    item = db.get(Item, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item


# 3) GET ALL with pagination (skip + limit) and total count
@router.get("", response_model=PaginatedItems)
def list_items(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
):
    logger.info("List items called (skip=%s, limit=%s)", skip, limit)
    total = db.scalar(select(func.count()).select_from(Item)) or 0
    items = db.scalars(select(Item).offset(skip).limit(limit)).all()
    return {"total": total, "skip": skip, "limit": limit, "items": items}


# 4) UPDATE (partial update)
@router.put("/{item_id}", response_model=ItemOut)
def update_item(item_id: int, payload: ItemUpdate, db: Session = Depends(get_db)):
    logger.info("Update item called: %s", item_id)
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
@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_item(item_id: int, db: Session = Depends(get_db)):
    logger.info("Delete item called: %s", item_id)
    item = db.get(Item, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    db.delete(item)
    db.commit()
    return None
