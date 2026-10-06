from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name : str = 'Aditya'
    age : Optional[int] = None
    email : EmailStr
    cgpa : float = Field(gt=0, lt=10)

new_student = {'age':24, 'email':'abcd@sfdg.com', 'cgpa':1}
student = Student(**new_student)

print(student)