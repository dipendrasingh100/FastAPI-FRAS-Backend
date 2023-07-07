from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import JSONResponse
from sqlalchemy import Table
from sqlalchemy.orm import Session
from user import models
from user.database import engine, SessionLocal
from face_recognition import load_image_file, face_encodings, compare_faces, face_locations
import json
import numpy as np
import cv2
app = FastAPI()

models.Base.metadata.create_all(engine)

metadata = models.Base.metadata
users_table = Table('users', metadata, autoload=True)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/register", status_code=status.HTTP_201_CREATED, tags=["Face Recognition Based Attendance System"])
def Register(name: str, email: str, file: UploadFile = File(None), db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == email).first()
    if db_user:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    image = load_image_file(file.file)
    boxes = face_locations(image, model='FaceNet')
    face_encoding = json.dumps(face_encodings(image, boxes)[0].tolist())

    face = db.query(models.User).filter(models.User.face_encoding == face_encoding).first()
    if face:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Face already registered")

    user = models.User(name=name, email=email, face_encoding=face_encoding)

    db.add(user)
    db.commit()
    db.refresh(user)
    return user


# @app.post("/recognise", status_code=status.HTTP_200_OK, tags=["Face Recognition Based Attendance System"])#prev
# def Recognise(file: UploadFile = File(None), db: Session = Depends(get_db)):
#     image = load_image_file(file.file)
#     boxes = face_locations(image,model='FaceNet')
#     face_encoding = face_encodings(image, boxes)
#     users = db.query(models.User).all()
#
#     for user in users:
#         user_face_encoding = np.array(json.loads(user.face_encoding))
#         if compare_faces(user_face_encoding, face_encoding)[0]:
#             return JSONResponse({"name": user.name, "email": user.email})
#     return JSONResponse({"message": "Face not recognised."})

@app.post("/recognise", status_code=status.HTTP_200_OK, tags=["Face Recognition Based Attendance System"])#me 1
def Recognise(file: UploadFile = File(None), db: Session = Depends(get_db)):
    image = load_image_file(file.file)
    boxes = face_locations(image, model='FaceNet')

    if not boxes:
        return JSONResponse({"message": "No face found in the image."})

    face_encoding = face_encodings(image, boxes)

    if not face_encoding:
        return JSONResponse({"message": "Face encoding failed. Unable to recognize faces."})

    users = db.query(models.User).all()

    for user in users:
        user_face_encoding = np.array(json.loads(user.face_encoding))
        if compare_faces(user_face_encoding, face_encoding)[0]:
            return JSONResponse({"name": user.name, "email": user.email})

    return JSONResponse({"message": "Face not recognised."})

# @app.post("/recognise", status_code=status.HTTP_200_OK, tags=["Face Recognition Based Attendance System"])
# def Recognise(file: UploadFile = File(None), db: Session = Depends(get_db)):
#     image = load_image_file(file.file)
#
#     # Preprocessing techniques
#     resized_image = cv2.resize(image, (1600, 1200))
#
#     # Example: Resizing the image
#     gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
#
#     # Adaptive thresholding
#     threshold_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
#
#     # Perform face detection on the preprocessed image
#     boxes = face_locations(image, model='FaceNet')
#     if not boxes:
#         boxes = face_locations(image, model='cnn')
#
#     if not boxes:
#         return JSONResponse({"message": "No face found in the image."})
#
#     face_encoding = face_encodings(resized_image, boxes)
#
#     if not face_encoding:
#         return JSONResponse({"message": "Face encoding failed. Unable to recognize faces."})
#
#     users = db.query(models.User).all()
#
#     for user in users:
#         user_face_encoding = np.array(json.loads(user.face_encoding))
#         if compare_faces(user_face_encoding, face_encoding)[0]:
#             return JSONResponse({"name": user.name, "email": user.email})
#
#     return JSONResponse({"message": "Face not recognized."})


# @app.post("/recognise", status_code=status.HTTP_200_OK, tags=["Face Recognition Based Attendance System"])#test
# def Recognise(file: UploadFile = File(None), db: Session = Depends(get_db)):
#     image = load_image_file(file.file)
#     boxes = face_locations(image, model='FaceNet')
#
#     if not boxes:
#         return JSONResponse({"message": "No face found in the image."})
#
#     # Optional: You can add a check for the face bounding box size to filter out small or low-quality faces.
#     # For example, you can define a minimum size threshold (in pixels) and remove faces with smaller bounding boxes.
#     min_face_size = 30  # Adjust the minimum face size as per your requirements
#     filtered_boxes = [box for box in boxes if (box[2] - box[0]) >= min_face_size and (box[3] - box[1]) >= min_face_size]
#
#     if not filtered_boxes:
#         return JSONResponse({"message": "No suitable face found in the image. Please provide a higher-quality image."})
#
#     face_encoding = face_encodings(image, filtered_boxes)
#
#     if not face_encoding:
#         return JSONResponse({"message": "Face encoding failed. Unable to recognize faces."})
#
#     users = db.query(models.User).all()
#
#     for user in users:
#         user_face_encoding = np.array(json.loads(user.face_encoding))
#         if compare_faces(user_face_encoding, face_encoding)[0]:
#             return JSONResponse({"name": user.name, "email": user.email})
#
#     return JSONResponse({"message": "Face not recognised."})


@app.get("/get_users", status_code=status.HTTP_200_OK, tags=["Face Recognition Based Attendance System"])
def all(db: Session = Depends(get_db)):
    user = db.query(models.User).all()
    if not user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No Record Found!")
    user = {u.id: [u.name, u.email] for u in user}
    return {"data": user}


@app.delete("/delete_user/{userid}", status_code=status.HTTP_200_OK, tags=["Face Recognition Based Attendance System"])
def delete(userid: int, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == userid).first()

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"User with id {userid} not found")
    db.delete(user)
    db.commit()
    return "Deleted"


@app.put("/update_user/{userid}", status_code=status.HTTP_200_OK, tags=["Face Recognition Based Attendance System"])
def update(userid: int, name: str, email: str, file: UploadFile = File(None), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == userid).first()

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"User with id {userid} not found")

    image = load_image_file(file.file)
    boxes = face_locations(image, model='FaceNet')
    face_encoding = json.dumps(face_encodings(image, boxes)[0].tolist())
    # Updating record
    user.name = name
    user.email = email
    user.face_encoding = face_encoding
    db.commit()
    return "Updated!"


@app.delete("/delete_records", status_code=status.HTTP_200_OK)
def drop(db:Session = Depends(get_db)):
    db.query(models.User).delete()
    db.commit()
    return {'message': 'User Records are deleted.'}
#
#
#
#
# from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
# from fastapi.responses import JSONResponse
# from sqlalchemy import Table
# from sqlalchemy.orm import Session
# from user import models
# from user.database import engine, SessionLocal
# from face_recognition import load_image_file, face_encodings, compare_faces, face_locations
# import json
# import numpy as np
# import cv2
#
# app = FastAPI()
#
# models.Base.metadata.create_all(engine)
#
# metadata = models.Base.metadata
# users_table = Table('users', metadata, autoload=True)
#
#
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()
#
#
# @app.post("/register", status_code=status.HTTP_201_CREATED, tags=["Face Recognition Based Attendance System"])
# def Register(name: str, email: str, file: UploadFile = File(None), db: Session = Depends(get_db)):
#     db_user = db.query(models.User).filter(models.User.email == email).first()
#     if db_user:
#         raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")
#
#     image = load_image_file(file.file)
#     boxes = face_locations(image, model='FaceNet')
#     face_encoding = json.dumps(face_encodings(image, boxes)[0].tolist())
#
#     face = db.query(models.User).filter(models.User.face_encoding == face_encoding).first()
#     if face:
#         raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Face already registered")
#
#     user = models.User(name=name, email=email, face_encoding=face_encoding)
#
#     db.add(user)
#     db.commit()
#     db.refresh(user)
#     return user
#
#
# @app.post("/recognise", status_code=status.HTTP_200_OK, tags=["Face Recognition Based Attendance System"])
# def Recognise(file: UploadFile = File(None), db: Session = Depends(get_db)):
#     image = load_image_file(file.file)
#
#     # Preprocessing techniques
#     resized_image = cv2.resize(image, (1600, 1200))
#     gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
#
#     # Perform face detection on the preprocessed image
#     boxes = face_locations(gray_image, model='hog')
#     if not boxes:
#         return JSONResponse({"message": "No face found in the image."})
#
#     face_encoding = face_encodings(resized_image, boxes)
#
#     if not face_encoding:
#         return JSONResponse({"message": "Face encoding failed. Unable to recognize faces."})
#
#     users = db.query(models.User).all()
#
#     for user in users:
#         user_face_encoding = np.array(json.loads(user.face_encoding))
#         if compare_faces(user_face_encoding, face_encoding)[0]:
#             return JSONResponse({"name": user.name, "email": user.email})
#
#     return JSONResponse({"message": "Face not recognised."})
#
#
# @app.get("/get_users", status_code=status.HTTP_200_OK, tags=["Face Recognition Based Attendance System"])
# def all(db: Session = Depends(get_db)):
#     user = db.query(models.User).all()
#     if not user:
#         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No Record Found!")
#     user = {u.id: [u.name, u.email] for u in user}
#     return {"data": user}
#
#
# @app.delete("/delete_user/{userid}", status_code=status.HTTP_200_OK, tags=["Face Recognition Based Attendance System"])
# def delete(userid: int, db: Session = Depends(get_db)):
#     user = db.query(models.User).filter(models.User.id == userid).first()
#
#     if not user:
#         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
#                             detail=f"User with id {userid} not found")
#     db.delete(user)
#     db.commit()
#     return "Deleted"
#
#
# @app.put("/update_user/{userid}", status_code=status.HTTP_200_OK, tags=["Face Recognition Based Attendance System"])
# def update(userid: int, name: str, email: str, file: UploadFile = File(None), db: Session = Depends(get_db)):
#     user = db.query(models.User).filter(models.User.id == userid).first()
#
#     if not user:
#         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
#                             detail=f"User with id {userid} not found")
#
#     image = load_image_file(file.file)
#     boxes = face_locations(image, model='FaceNet')
#     face_encoding = json.dumps(face_encodings(image, boxes)[0].tolist())
#     # Updating record
#     user.name = name
#     user.email = email
#     user.face_encoding = face_encoding
#     db.commit()
#     return "Updated!"
#
#
# @app.delete("/delete_records", status_code=status.HTTP_200_OK)
# def drop(db:Session = Depends(get_db)):
#     db.query(models.User).delete()
#     db.commit()
#     return {'message': 'User Records are deleted.'}
