from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import update, and_, ForeignKey, CheckConstraint
import os
import datetime as dt

# this_database = "ml_db"
this_database = "mltestdev"

# Enviroment options
host = os.environ.get("HOST")
database = this_database
user = os.environ.get("USER")
password = os.environ.get("PW")
port = os.environ.get("PORT")

engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{database}")
Session = sessionmaker(bind=engine)
sess = Session()

Base = declarative_base()

"""
Classes
"""


# users table
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)


# models table
class Model(Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True)
    modelname = Column(String(50), unique=True, nullable=False)
    status = Column(String(1), default="B")
    created_date = Column(DateTime)
    updated_date = Column(DateTime)


# frequency
class Frequency(Base):
    __tablename__ = "frequency"
    id = Column(Integer, primary_key=True)
    model = Column(Integer, ForeignKey("models.id"))
    computer_name = Column(String(50), nullable=False)
    frequency = Column(Float)


# WorkingDays
class WorkingDays(Base):
    __tablename__ = "working_days"
    id = Column(Integer, primary_key=True)
    userid = Column(Integer, ForeignKey("users.id"))
    monday = Column(Float)
    tuesday = Column(Float)
    wednesday = Column(Float)
    thursday = Column(Float)
    friday = Column(Float)
    saturday = Column(Float)
    sunday = Column(Float)


# Work_hours
class WorkHours(Base):
    __tablename__ = "work_hours"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    mon_strt = Column(Integer, CheckConstraint("mon_strt >= 0 AND mon_strt <= 23"))
    mon_end = Column(Integer, CheckConstraint("mon_end >= 0 AND mon_end <= 23"))
    tues_strt = Column(Integer, CheckConstraint("tues_strt >= 0 AND tues_strt <= 23"))
    tues_end = Column(Integer, CheckConstraint("tues_end >= 0 AND tues_end <= 23"))
    wed_strt = Column(Integer, CheckConstraint("wed_strt >= 0 AND wed_strt <= 23"))
    wed_end = Column(Integer, CheckConstraint("wed_end >= 0 AND wed_end <= 23"))
    thur_strt = Column(Integer, CheckConstraint("thur_strt >= 0 AND thur_strt <= 23"))
    thur_end = Column(Integer, CheckConstraint("thur_end >= 0 AND thur_end <= 23"))
    fri_strt = Column(Integer, CheckConstraint("fri_strt >= 0 AND fri_strt <= 23"))
    fri_end = Column(Integer, CheckConstraint("fri_end >= 0 AND fri_end <= 23"))
    sat_strt = Column(Integer, CheckConstraint("sat_strt >= 0 AND sat_strt <= 23"))
    sat_end = Column(Integer, CheckConstraint("sat_end >= 0 AND sat_end <= 23"))
    sun_strt = Column(Integer, CheckConstraint("sun_strt >= 0 AND sun_strt <= 23"))
    sun_end = Column(Integer, CheckConstraint("sun_end >= 0 AND sun_end <= 23"))


# departments table
class Department(Base):
    __tablename__ = "departments"
    id = Column(Integer, primary_key=True)
    deptname = Column(String(50), unique=True, nullable=False)
    association = Column(Integer)


# user_model table
class UserModel(Base):
    __tablename__ = "user_model"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    model_id = Column(Integer, ForeignKey("models.id"))


# user_dept table
class UserDept(Base):
    __tablename__ = "user_dept"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    dept_id = Column(Integer, ForeignKey("departments.id"))


# logged table
class Logged(Base):
    __tablename__ = "logged"
    id = Column(Integer, primary_key=True)
    created = Column(DateTime)
    level = Column(Integer, CheckConstraint("level >= 0 AND level <= 4"))
    log_data = Column(String)
    log_source = Column(String)
    log_notes = Column(String)
    wb = Column(String)


"""
Methods for retrieving table data
"""
"""
C - Create (insert)
"""


def insert_user(name):
    """
    insert_user()
    : Add a NEW user to the database :
    Input:
    ------
    name - a 32 character hexidecimal value
    - should be unique otherwise will not be allowed to insert
    Output:
    -------
    id - the id generated for the new user
    """

    with sess:
        in_user = User(username=name)
        sess.add(in_user)
        sess.commit()
        if in_user.id is None:
            sess.refresh(in_user)
        return in_user.id


def insert_model(in_name, in_date):
    """
    insert_model()
    : Adds a NEW model to the database :
    Input:
    in_name: a 32 character hexidecomal value
    - should be unique otherwise will not allow insert
    in_date - date of created record in peroper format 'YYYY-MM-DD'
    Note: update_date will use that same data when being created

    Output:
    -------
    id - the new id assigned to the model
    """

    with sess:
        in_model = Model(
            modelname=in_name,
            status="B",
            created_date=in_date,
            updated_date=in_date,
        )
        sess.add(in_model)
        sess.commit()
        if in_model.id is None:
            in_model.refresh()

        return in_model.id


def insert_frequency(in_name, in_computer, freq):
    """
    insert_frequency()
    : Adds a NEW frequency record to the database :
    Input:
    ------
    in_name - if a valid modelname will be looked up for the id,
    otherwise the id of the associated model
    in_computer - a 32 character hexidecimal name
    freq - a float value

    Output:
    -------
    Returns the record id
    """

    with sess:
        in_frequency = Frequency(
            model=in_name, computer_name=in_computer, frequency=freq
        )
        sess.add(in_frequency)
        sess.commit()
        if in_frequency.id is None:
            in_frequency.refresh()

        return in_frequency.id


def insert_w_days(in_name, in_M, in_Tu, in_W, in_Th, in_F, in_Sat, in_Sun):
    """
    insert_w_days()
    : Adds a NEW workings_days record to the database :

    Input:
    in_name : the username  or user id associated with the record
    in_M, in_Tu .. in_Sun : the weekly record Monday thru Sunday

    Output:
    -------
    The id of the record created
    """

    with sess:
        in_wdays = WorkingDays(
            userid=in_name,
            monday=in_M,
            tuesday=in_Tu,
            wednesday=in_W,
            thursday=in_Th,
            friday=in_F,
            saturday=in_Sat,
            sunday=in_Sun,
        )
        sess.add(in_wdays)
        sess.commit()
        if in_wdays.id is None:
            in_wdays.refresh()

        return in_wdays.id


def insert_work_hours(
    in_user,
    mon_s,
    mon_e,
    tues_s,
    tues_e,
    wed_s,
    wed_e,
    thur_s,
    thur_e,
    fri_s,
    fri_e,
    sat_s,
    sat_e,
    sun_s,
    sun_e,
):
    """
    insert_work_hours()
    : Adds a new work_hours record to the database :
    Inout:
    ------
    in_user - the usdername or id associated with the record
    mon_s, mon_e ... sun_s, sun_e - start and end times for each day,
    - must be an integer between 0 and 23

    Output:
    -------
    id : the id of the the record

    """
    with sess:
        in_hours = WorkHours(
            user_id=in_user,
            mon_strt=mon_s,
            mon_end=mon_e,
            tues_strt=tues_s,
            tues_end=tues_e,
            wed_strt=wed_s,
            wed_end=wed_e,
            thur_strt=thur_s,
            thur_end=thur_e,
            fri_strt=fri_s,
            fri_end=fri_e,
            sat_strt=sat_s,
            sat_end=sat_e,
            sun_strt=sun_s,
            sun_end=sun_e,
        )
        sess.add(in_hours)
        sess.commit()
        if in_hours.id is None:
            in_hours.refresh()

        return in_hours.id


def insert_department(name, assoc):
    """
    insert_department()
    : Adds a NEW department record to the database :

    Input:
    ------
    name - the department name to be added
        (shortened forms will remain shortened)
    assoc: The department to which this department reports
    - Allowing for a heirachical search wehen implemented

    Output:
    -------
    id - the id of the record - Assoc needs to use this id as a reference.
    """

    with sess:
        in_dept = Department(deptname=name, association=assoc)
        sess.add(in_dept)
        sess.commit()
        if in_dept.id is None:
            in_dept.refresh()

        return in_dept.id


def insert_user_model(uid, mid):
    """
    insert_user_model()
    : Adds a NEW user_model record to the database :
    - user_model is an inverse table associating users with models

    Input:
    ------
    uid - The user id
    mid - the model id
    -- These ids are of users and models in thier own tables

    Output:
    -------
    No output returned
    """

    with sess:
        in_um = UserModel(user_id=uid, model_id=mid)
        sess.add(in_um)
        sess.commit()
        return


def insert_user_dept(uid, did):
    """
    insert_user_dept()
    : Adds a NEW user_dept record to the database :
    - user_dept is an inverse table associating users with departments

    Input:
    ------
    uid - The user id
    did - the department id
    -- These ids are of users and deprtments in thier own tables

    Output:
    -------
    No output returned
    """
    with sess:
        in_ud = UserDept(user_id=uid, dept_id=did)
        sess.add(in_ud)
        sess.commit()
        return


def insert_log(cdate, lev, ldata, lsource, lnotes, host):
    """
    insert_log()
    : Writes a log record to the database :

    Input:
    ------
    cdate -  the created_date of the record: format YYYY-MM-DD
    lev - The level, can be 0 - 4, 1 is good, 0 is a debbugging comment,
    2 - 4 levels of increasing concern
    ldata - The requisite log message
    lsource - The document, and function, if available where the log comes from
    lnotes - line numbers, etc.
    host - the workbench where the log originated
    """
    with sess:
        in_log = Logged(
            created=cdate,
            level=lev,
            log_data=ldata,
            log_source=lsource,
            log_notes=lnotes,
            wb=host,
        )
        sess.add(in_log)
        sess.commit()


"""
R - Retrieve (get)
"""


# Users
def get_users():
    """
    get_users()
    : Get all user data from the database :

    Input:
    ------
    None

    Output:
    -------
    A list of User objects
    elements: .id, .username
    """
    with sess:
        return sess.query(User).all()


def get_user_id(in_name):
    """
    get_user_id()
    : Retrieves the id of a user given the username :

    Input:
    ------
    in_name : a 32 character hexidecimal string

    Output:
    -------
    The expected id; if it exists
    -1, if the record cannot be found
    """
    with sess:
        res = sess.query(User).filter(User.username == in_name).all()
        if res == []:
            return -1
        else:
            return res[0].id


def get_user_name(in_id):
    """
    get_user_name()
    : Retrieves the username, based on the input id :

    Input:
    ------
    in_id - the id (integer value) being passed in

    Output:
    -------
    The username, if it exists
    [] - if not found
    """
    with sess:
        res = sess.query(User).filter(User.id == in_id)
        if not res:
            return []
        return res[0].username


def test_user_id(in_id):
    """
    test_user_id()
    : Validates a user id :

    Input:
    ------
    the id being tested

    Output:
    -------
    True if found, else False
    """
    with sess:
        res = sess.query(User).filter(User.id == in_id).all()

        if res == []:
            return False
        return True


# models
def get_models():
    """
    get_models()
    : Retrieves all models from the database :

    Input:
    ------
    None

    Output:
    -------
    A list of model objects,
    elements: .id, .status, .created_date, .updated_date
    """
    with sess:
        return sess.query(Model).all()


def get_models_by_status(in_stat):
    """
    get_models_by_status()
    : Brings up a set of model objects with a sp[ecific status :

    Input:
    ------
    The status - should be 'G', 'B' or 'U'

    Output:
    -------
    A list of model objects,
    elements: .id, .status, .created_date, .updated_date
    """

    with sess:
        return sess.query(Model).filter(Model.status == in_stat).all()


def get_model_id(in_name):
    """
    get_model_id()
    : Retrieves an id for a given modelname :

    Input:
    in_name - the modelname for which the id is being fetched

    Output:
    -------
    The id for the input modelname, if it exosts
    -1 , if nothing is found
    """

    with sess:
        res = sess.query(Model).filter(Model.modelname == in_name).all()
        if res == []:
            return -1
        my_Model = res[0]
        return my_Model.id


def get_model_name(in_id):
    """
    get_model_name()
    : Retrieves the modelname for the given input :
    Input:
    in_id: a valid model od (integer value)

    Output:
    -------
    The modelname associated with the input
    [] - if nothing is found
    """

    with sess:
        res = sess.query(Model).filter(Model.id == in_id).all()
        if res == []:
            return None

        return res[0].modelname


def test_model_id(in_id):
    """
    test_model_id()
    : Tests if the id is valid :

    Input:
    ------
    in_id - the id (integer value) to be tested for

    Output:
    -------
    True, if the id is valid
    False, if not
    """

    with sess:
        res = sess.query(Model).filter(Model.id == in_id).all()
        if res == []:
            return False
        return True


def test_frequency_model(in_model):
    with sess:
        f_set = sess.query(Frequency).filter(Frequency.model == in_model).all()
    return len(f_set) >= 1


# Frequency
def get_frequency(in_model):
    with sess:
        freq = []
        f_set = sess.query(Frequency).filter(Frequency.model == in_model).all()
        if f_set == []:
            return freq
        for item in f_set:
            freq.append((item.computer_name, item.frequency))

        return freq


# Work_days
def test_working_days_user(in_user):
    wd_set = sess.query(WorkingDays).filter(WorkingDays.userid == in_user).all()
    return len(wd_set) >= 1


# Work_days
def get_working_days(in_user):
    with sess:
        Work_days = []
        wd_set = sess.query(WorkingDays).filter(WorkingDays.userid == in_user).all()
        if wd_set == []:
            return Work_days
        Work_days.append(wd_set[0].monday)
        Work_days.append(wd_set[0].tuesday)
        Work_days.append(wd_set[0].wednesday)
        Work_days.append(wd_set[0].thursday)
        Work_days.append(wd_set[0].friday)
        Work_days.append(wd_set[0].saturday)
        Work_days.append(wd_set[0].sunday)
        return Work_days


# Work_hours
def test_work_hours_user(in_user):
    wh_set = sess.query(WorkHours).filter(WorkHours.user_id == in_user).all()
    return len(wh_set) >= 1


# Work_hours
def get_work_hours(in_user):
    with sess:
        Work_hours = []
        wh_set = sess.query(WorkHours).filter(WorkHours.user_id == in_user).all()
        if wh_set == []:
            return Work_hours
        Work_hours.append(wh_set[0].mon_strt)
        Work_hours.append(wh_set[0].mon_end)
        Work_hours.append(wh_set[0].tues_strt)
        Work_hours.append(wh_set[0].tues_end)
        Work_hours.append(wh_set[0].wed_strt)
        Work_hours.append(wh_set[0].wed_end)
        Work_hours.append(wh_set[0].thur_strt)
        Work_hours.append(wh_set[0].thur_end)
        Work_hours.append(wh_set[0].fri_strt)
        Work_hours.append(wh_set[0].fri_end)
        Work_hours.append(wh_set[0].sat_strt)
        Work_hours.append(wh_set[0].sat_end)
        Work_hours.append(wh_set[0].sun_strt)
        Work_hours.append(wh_set[0].sun_end)
        return Work_hours


# departments
def get_departments():
    with sess:
        return sess.query(Department).all()


def get_dept_id(dept_name):
    with sess:
        res = sess.query(Department).filter(Department.deptname == dept_name).all()
        if res == []:
            return -1

        return res[0].id


def get_dept_name(dept_id):
    with sess:
        res = sess.query(Department).filter(Department.id == dept_id).all()
        if res == []:
            return None

        return res[0].deptname


def test_dept_id(in_id):
    with sess:
        res = sess.query(Department).filter(Department.id == in_id).all()
        if res == []:
            return False
        return True


def get_dept_assoc(in_dept):
    with sess:
        res = None
        if isinstance(in_dept, str):
            res = sess.query(Department).filter(Department.deptname == in_dept).all()
        else:
            res = (
                sess.query(Department.association)
                .filter(Department.id == in_dept)
                .all()
            )
        if res == []:
            return -1
        return res[0].association


def get_user_models():
    with sess:
        return sess.query(UserModel).all()


def get_user_model_id(user_id, model_id):
    with sess:
        res = (
            sess.query(UserModel)
            .filter(and_(UserModel.user_id == user_id, UserModel.model_id == model_id))
            .all()
        )
        if res == []:
            return -1
        return res[0].id


def get_user_depts():
    with sess:
        return sess.query(UserDept).all()


def get_user_dept_id(user_id, dept_id):
    with sess:
        res = (
            sess.query(UserDept)
            .filter(and_(UserDept.user_id == user_id, UserDept.dept_id == dept_id))
            .all()
        )
        if res == []:
            return -1
        return res[0].id


"""
U - Update
"""


# users
def update_user_name(in_u, n_st):
    with sess:
        stmt = ""
        if isinstance(in_u, str):
            stmt = update(User).where(User.username == in_u).values(username=n_st)
        else:
            stmt = update(User).where(User.id == in_u).values(username=n_st)
        sess.execute(stmt)
        sess.commit()
    return


# models
def update_model_name(in_m, n_st):
    with sess:
        stmt = ""
        if isinstance(in_m, str):
            stmt = update(Model).where(Model.modelname == in_m).values(modelname=n_st)
        else:
            stmt = update(Model).where(Model.id == in_m).values(modelname=n_st)
        sess.execute(stmt)
        sess.commit()
    return


def update_model_status(in_m, n_st):
    with sess:
        stmt = ""
        if isinstance(in_m, str):
            stmt = update(Model).where(Model.modelname == in_m).values(status=n_st)
        else:
            stmt = update(Model).where(Model.id == in_m).values(status=n_st)
        sess.execute(stmt)
        sess.commit()
    return


def update_model_date(in_m, n_st):
    with sess:
        stmt = ""
        if isinstance(in_m, str):
            stmt = (
                update(Model).where(Model.modelname == in_m).values(created_date=n_st)
            )
        else:
            stmt = update(Model).where(Model.id == in_m).values(created_date=n_st)
        sess.execute(stmt)
        sess.commit()
    return


# departments
def update_dept_name(in_d, n_st):
    with sess:
        stmt = ""
        if isinstance(in_d, str):
            stmt = (
                update(Department)
                .where(Department.deptname == in_d)
                .values(deptname=n_st)
            )
        else:
            stmt = update(Department).where(Department.id == in_d).values(deptname=n_st)
        sess.execute(stmt)
        sess.commit()
    return


def update_w_days(in_name, in_M, in_Tu, in_W, in_Th, in_F, in_Sat, in_Sun):
    """
    update_w_days()
    : update an existing workings_days record to the database :

    Input:
    in_name : the username  or user id associated with the record
    in_M, in_Tu .. in_Sun : the weekly record Monday thru Sunday
    """

    with sess:
        stmt = (
            update(WorkingDays)
            .where(WorkingDays.userid == in_name)
            .values(
                monday=in_M,
                tuesday=in_Tu,
                wednesday=in_W,
                thursday=in_Th,
                friday=in_F,
                saturday=in_Sat,
                sunday=in_Sun,
            )
        )
        sess.execute(stmt)
        sess.commit()


def update_work_hours(
    in_user,
    mon_s,
    mon_e,
    tues_s,
    tues_e,
    wed_s,
    wed_e,
    thur_s,
    thur_e,
    fri_s,
    fri_e,
    sat_s,
    sat_e,
    sun_s,
    sun_e,
):
    """
    update_work_hours()
    : Updates an existing work_hours record to the database :
    Inout:
    ------
    in_user - the username or id associated with the record
    mon_s, mon_e ... sun_s, sun_e - start and end times for each day
    - must be an integer between 0 and 23

    """
    with sess:
        stmt = (
            update(WorkHours)
            .where(WorkHours.user_id == in_user)
            .values(
                mon_strt=mon_s,
                mon_end=mon_e,
                tues_strt=tues_s,
                tues_end=tues_e,
                wed_strt=wed_s,
                wed_end=wed_e,
                thur_strt=thur_s,
                thur_end=thur_e,
                fri_strt=fri_s,
                fri_end=fri_e,
                sat_strt=sat_s,
                sat_end=sat_e,
                sun_strt=sun_s,
                sun_end=sun_e,
            )
        )
        sess.execute(stmt)


def update_department_assoc(in_id, assoc):
    with sess:
        stmt = ""
        stmt = (
            update(Department).where(Department.id == in_id).values(association=assoc)
        )
        sess.execute(stmt)
        sess.commit()
        if in_id is None:
            in_id.refresh()

        return in_id


"""
- R - Remove
"""


def delete_frequency_records(in_model):
    """
    delete_frequency_records()

    Input:
    in_model - the id of the model for which all records (i.e. all machines)
    will be deleted

    Output:
    ret - 1 - Okay, 0 - Problem
    """

    with sess:
        sess.query(Frequency).filter(Frequency.model == in_model).delete()
        sess.commit()


def delete_logged_records():
    """
    Delete logged records older than the specified threshold.

    This function removes all logs that were created more than 2 weeks
    before the current date to maintain database efficiency.

    Notes of Usage:
        1. Call directly when log cleanup is needed
        2. Can be integrated with MLDB.py by passing the current date
    """
    today = dt.datetime.now()
    threshold_date = today - dt.timedelta(weeks=2)
    with sess:
        sess.query(Logged).filter(Logged.created < str(threshold_date)).delete()
        sess.commit()  # Ensure changes are committed to database


# if __name__ == "__main__":
#     try:
#         Base.metadata.create_all(engine)
#     except Exception as error:
#         print("The tables were not created.")
#         print(error)


""" End of Module sqlutils """
