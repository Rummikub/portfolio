"""
Module: MLDB
Function: Act as API to the Database

Conventions:
Retrieving data - names begin with "get"
Saving data - names begin with either "new" for new records or "update" if
amending a record
Deleting data - names begin with "remove"

Tables in use:
users - stores user data
models - stores model data
departments - stores Department options to associate with users.

Note: at the time of setting up, users and models both have the same name
however, as this reaches more users, this may change and as such, we use both
usernames and model names independantly.
"""

# For DB access
import sqlutils as su
import datetime as dt
import os
import inspect
import pytz

# Production
# this_database = "ml_db"
# Testing, Experimenting and Development
this_database = "mltestdev"

# Saves current date into SQL appropriate format
d_today = dt.datetime.now(pytz.timezone("America/New_York")).strftime(
    "%Y-%m-%d %H:%M:%S"
)

# A general message
bad_connect = "Connection not available:"
# This list can be amended as required to always contain the list of tables
# used other than "logged"
table_list = ["users", "models", "departments"]
default_dept_assoc = 3
p_fields = ["id", "username", "modelname", "deptname", "status", "frequency"]
log_text_file = "../data/logging_errors.txt"
model_status = ["G", "B", "U"]


def get_workbench():
    notebook_str = os.environ.get("HOSTNAME")
    return f"wb {notebook_str[-3]}"


def lineno():
    """
    Returns the current line number in our program.
    """

    my_line = inspect.currentframe().f_back.f_lineno
    out_str = f"@ line {my_line}"
    return out_str


def write_log(log_msg, level, source, notes):
    """
    write_log()
    - writes a record to the logged data table
    Inputs:
    - log_msg (str) - the message being sent
    - level - an integer from 0 to 4
    -- 0 - Debug (Probably not used here)
    -- 1 - Info - simple information
    -- 2 - Warning - not an error, but worth reporting
    -- 3 - Error - An actual error has been accounted for
    -- 4 - Critical - Very Serious Problem (Probably not used here)
    log_notes - additional information
    """

    if not isinstance(level, int):
        write_log(
            "Level given is not an integer value",
            3,
            "MLDB.py",
            f" write_log: {log_msg}",
        )
        return
    else:
        good_level = level in [0, 1, 2, 3, 4]
        if not good_level:
            bad_level = "Level " + str(level) + " is not acceptable"
            write_log(bad_level, 3, "MLDB.py", f"write_log: {log_msg}")
            return
    try:
        su.insert_log(d_today, level, log_msg, source, notes, get_workbench())
    except Exception as e:
        err_str = f"{format(e)} - could not log msg: {log_msg} from {source}\n"
        err_str = err_str + f"From {get_workbench()}"
        write_text_log(err_str)
    return


def write_text_log(msg):
    """
    write_text_log()
    In the event that a write_log (to the database) fails, a line
    of text will be added to the error_logging.txt file found in
    the ../data subfolder
    """
    if os.path.exists(log_text_file):
        with open(log_text_file, "a") as w_file:
            w_file.write(msg + "\n")
    else:
        with open(log_text_file, "w") as w_file:
            w_file.write(msg + "\n")
    return


def convert_sql_result(res):
    """
    convert_sql_result()
    Onput - res - result of a fetchall() call
    Output:
    - nlist: transformation to a list
    """
    if res is None:
        return []

    nlist = []
    for t in res:
        nlist.append(t[0])

    return nlist


"""
General usage functions
"""


def is_valid_id(in_table, in_id):
    """
    is_valid_id(in_table, in_ID)
    Function to test is the id provided is a valid id based on column
    in_table: valid table
    return a boolean value(valid id
    """

    # defaults to False - Valid format (numeric)
    v_format = False
    # Default value for is in usee
    in_use = False
    v_format = isinstance(in_id, int)
    if not v_format:
        return False

    # test for valid in_table
    if in_table not in table_list:
        return False

    if in_table == "users":
        in_use = su.test_user_id(in_id)
    elif in_table == "models":
        in_use = su.test_model_id(in_id)
    else:
        in_use = su.test_dept_id(in_id)

    return in_use


def is_hexstring(in_string):
    """
    Input:
    - string to be tested
    Hex_string(in_string)
    reuturns:
    True if provided string is in Hexadecimal format
    """
    if len(in_string) == 0:
        return False
    try:
        int(in_string, 16)
        return True
    except ValueError:
        return False


def is_valid_name(in_name, in_length):
    """
    Meant for internal use by this module
    is_valid_username(in_name)
    Function to test if username is in a valid format, and if used already
    Inputs:
    in_name - string to be tested
    in_lrength - expected length of string
    output: True if in_name is valid
    """
    # Test for valid input
    if in_name == "":
        return False
    elif len(in_name) != in_length:
        return False
    return is_hexstring(in_name)


def pair_exists(table_comp, user_id, comp_id):
    """
    Inputs:
    table_comp - the table compared to against the user table
    user_id - the user's id - at this time all inverse tables include the
    users table ids
    comp_id - the compared to table's id
    Returns:
    True, if a pair of the two IDs actually exists in the table
    This is used to test a pair of IDs in the inverse tables
    Currently it tests:
    user_model
    user_dept

    Should more such tables be created, this needs to be recorded appropriately
    """
    res = []
    if table_comp == "models":
        res = su.get_user_model_id(user_id, comp_id)
    else:
        res = su.get_user_dept_id(user_id, comp_id)

    return res != -1


def get_id(in_table, in_name):
    """
    Retrieve the id of a given name for a given table
    get_id()
    Inputs:
    in_table - Table the id belongs to
    in_name - name associated with the id
    output: user_id, the integer value
    """

    if in_table not in table_list:
        return -1

    try:
        res = 0
        if in_table == "users":
            res = su.get_user_id(in_name)
        elif in_table == "models":
            res = su.get_model_id(in_name)
        else:
            res = su.get_dept_id(in_name)
        print(type(res))
        return res
    except Exception as e:
        err_msg = f"Problem retrieving {in_table} id\n{format(e)}"
        write_log(err_msg, 3, "MLDB.py", "get_id()")
        return -1, err_msg


def update_name(in_table, in_id, new_name):
    """
    update_name(in_table, in_id, new_name)
    Input:
    in_table - the table that is to be updated (users, models, or departments)
    in_id    - a valid integer that matches a name that is to be updated
    new_name - a valid username that is associated with the id
    Returns:
    - 1 if successful, 0 if not
    msg - for log putposes
    """

    l_source = "MLDB.py"
    l_notes = "update_name()"

    # Test for valid table
    if in_table not in table_list:
        write_log(f"Invalid table: {in_table}", 2, l_source, lineno())
        return 0, "Invalid table name"

    # ret_str = ""
    is_good = is_valid_id(in_table, in_id)

    if not is_good:
        write_log(f"Invalid id provided: {in_id}", 2, l_source, l_notes + lineno())
        return 0, "Invalid id provided"

    in_use = False
    # test for used user name
    try:
        if in_table == "users":
            in_use = su.get_user_id(new_name) != -1
        elif in_table == "models":
            in_use = su.get_model_id(new_name) != -1
        else:
            in_use = su.get_dept_id(new_name) != -1

        if in_use:
            ret_msg = "name: '" + new_name + "' cannot be used to update '"
            ret_msg = ret_msg + in_table + "' as it is already in use"
            write_log(ret_msg, 2, l_source.l_notes + lineno())
            return 0, ret_msg

        old_name = ""
        if in_table == "users":
            old_name = su.get_user_name(in_id)
        elif in_table == "models":
            old_name = su.get_model_name(in_id)
        else:
            old_name = su.get_dept_name(in_id)

        if in_table == "users":
            su.update_user_name(in_id, new_name)
        elif in_table == "models":
            su.update_model_name(in_id, new_name)
        elif in_table == "departments":
            su.update_dept_name(in_id, new_name)
        else:
            print("Incorrect table name provided!")

    except Exception as e:
        err_msg = f"Issue updating name: {format(e)}"
        write_log(err_msg, 2, l_source, l_notes + lineno())
        return 0, err_msg + lineno()

    ret_msg = f"{in_table}'s "
    if in_table == "users":
        ret_msg = ret_msg + "usernasme "
    elif in_table == "models":
        ret_msg = ret_msg + "modelname "
    else:
        ret_msg = ret_msg + "deptname "
    ret_msg = ret_msg + f"was updated from '{old_name}' to '{new_name}'"
    write_log(ret_msg, 1, l_source, "")
    return 1, ret_msg


def convert_list_to_string(in_list):
    """
    convert_list_to_string()
        Takes a list and turns it into a comma delineated string
        Input:
        - in_list - the list to be deconstructed
        Output:
        - res - 1 if successful, 0 - otherwise
        - outstr - the resultant string value
    """
    outstr = ""
    # test if input is a string
    if isinstance(in_list, str):
        return 1, in_list
    # test for incorrect input
    if not isinstance(in_list, list):
        if isinstance(in_list, tuple):
            in_list = list(in_list)
        else:
            return 0, "Not convertable"
    # test for empty list
    if len(in_list) < 1:
        return 1, ""
    for item in in_list:
        if not isinstance(item, str):
            if isinstance(item, list) or isinstance(item, tuple):
                res, out_str = convert_list_to_string(item)
                if res == 0:
                    return 0, "Not convertable"
                else:
                    if isinstance(out_str, str):
                        item = out_str
                    else:
                        res, out_str = convert_list_to_string(item)
            elif isinstance(item, int):
                item = str(item)

        # at this point we should have only strings
        if outstr != "":
            outstr = outstr + ", "
        outstr = outstr + item

    return 1, outstr


def valid_val(x, v):
    try:
        getattr(x, v)
    except AttributeError:
        return False
    return True


def process_list(res_set, fields):
    """
    Takes the result set and turn it into a list for use elsewhere
    """

    # list of fields that could be called
    out_list = []

    # test for length of fields
    if len(fields) == 1:
        for item in res_set:
            lval = getattr(item, fields[0])
            out_list.append(lval)
        return out_list
    elif len(fields) > 1:
        # multi value
        for item in res_set:
            inner = []
            for y in range(len(p_fields)):
                if valid_val(item, p_fields[y - 1]):
                    if p_fields[y - 1] in fields:
                        vv = getattr(item, p_fields[y - 1])
                        inner.append(vv)
            out_list.append(list(inner))
        return out_list
    else:
        # nothing found
        return []


def get_all(table, fields):
    """
    get_all()
        retrieves data based on table and fields
    Input:
        - Table - a viable table name, based on table_list
        - fields - one or more fields to select into output
    Output:
        - Result 1 if good, else 0
        - List of output fields, if more than one selected, a list of lists
    """
    # test for legitimate table
    if not (table in table_list):
        return 0, []

    # make sure that fields is a list
    if not isinstance(fields, list):
        fields = fields.split(",")
    res = None
    if table == "users":
        res = su.get_users()
    elif table == "models":
        res = su.get_models()
    else:
        # default to departments
        res = su.get_departments()
    return process_list(res, fields)


def get_name(in_table, in_id):
    """
    get_name()
    retrieves a name based on a table, given an id
    Input:
    in_table -  name of table
    in_id - id in table of name
    Returns:
    1 if okay, 0 if not
    out_name - name retrieved
    """

    out_name = ""
    try:
        if in_table in table_list:
            if in_table == "users":
                out_name = su.get_user_name(in_id)
            elif in_table == "models":
                out_name = su.get_model_name(in_id)
            else:
                out_name = su.get_dept_name(in_id)
            return 1, out_name
        else:
            return 0, ""
    except Exception as e:
        err_msg = f"Problem retrieving name {format(e)}"
        write_log(err_msg, 3, "MLDB.py", f"get_name() {lineno()}")
    return 0, ""


# ===================
# User manipulation
# ===================


def new_user(user, model, dept):
    """
    new_user()
    Input:
    - user - a string value in Hexidecimal format and 32 characters long
    - model -  a string value in Hexidecimal format and 32 char long,
    may be same as user, if '', then presumed same as user
    - dept - a string value, if left '',
    will default to 'AO': Accountability Office
    Returns:
    int" 1 if succesful, 0 - if not
    A string which can be used for a log entry as provided
    """
    # test incoming data
    if not is_valid_name(user, 32):
        ret_str = "invalid user name: " + user
        write_log(ret_str, 2, "MLDB.py, new_user()", lineno())
        return 0, ret_str

    if model == "":
        model = user
    if not is_valid_name(model, 32):
        ret_str = f"Invalid model name: {model}"
        write_log(ret_str, 2, "MLDB.py, new_user()", "")
        return 0, ret_str

    if dept == "":
        dept = "AO"

    try:
        # retrieve ids, if -1, is not in database
        user_id = su.get_user_id(user)
        model_id = su.get_model_id(model)
        dept_id = su.get_dept_id(dept)

        # str for return with no issues
        ret_str = ""

        # insert new user and retrieve id, or retrieve existing id
        if user_id == -1:
            user_id = su.insert_user(user)
            ret_str = ret_str + f"New user: {user} added\n"
        else:
            ret_str = ret_str + f"User: {user} already exists\n"
            # insert new model

        if model_id == -1:
            model_id = su.insert_model(model, d_today)
            ret_str = ret_str + f"Model: {model} added\n"
        else:
            ret_str = ret_str + f"Model: {model} already exists\n"

        # insert new department
        if dept_id == -1:
            dept_id = su.insert_department(dept, default_dept_assoc)
            ret_str = ret_str + f"Department: {dept} added"
        else:
            ret_str = ret_str + f"Department: {dept} already exists"

        # Associate user and model
        if not pair_exists("models", user_id, model_id):
            su.insert_user_model(user_id, model_id)

        # Associate user with department
        if not pair_exists("dept", user_id, dept_id):
            su.insert_user_dept(user_id, dept_id)

        # Closing the database
        write_log(ret_str, 1, "new_user()", str(d_today))
    except Exception as e:
        err_msg = f"Failure in adding new user: {format(e)}"
        write_log(err_msg, 2, "MLDB.py", "new_user()")
        return 0, err_msg
    return 1, ret_str


"""
Model specific functions
"""


def update_model(model, in_val, item):
    """
    update_model(model, in_val, item)\n
    Takes the id or name, and updates the status (or other column)
        if added in the future
    Input:
    - Model: may be an int or a string
    -- must be valid values
    - item: either status (may add to this if required)
    Output:
    - 0 - if successful, 1 if not
    - msg: for logging purposes
    """
    try:
        if not isinstance(model, int):
            model = get_id("models", model)
        # chack for 'model' input voracity
        in_use = is_valid_id("models", model)

        if not in_use:  # might have passed a modelname
            GoodToGo = is_valid_name(model, 32)
            if not GoodToGo:
                err_str = (
                    "Input not valid, requires an id or name found in the database"
                )
                write_log(err_str, 2, "MLDB.py", f"update_model() {lineno()}")
                return 0, err_str
        if item == "status":
            if in_val not in model_status:
                err_str = "Invalid status: " + item + ", must be B, G or U"
                write_log(err_str, 2, "MLDB.py", f"update_model() {lineno}")
            su.update_model_status(model, in_val)
            su.update_model_date(model, d_today)
        else:
            err_str = "Unclear what '" + item + "' is: " + item
            write_log(err_str, 2, "MLDB.py", f" update_model() {lineno}")
            return 0, err_str
        # at this stage, we have a valid input
    except Exception as e:
        err_str = f"Problem updating model {format(e)}"
        write_log(err_str, 3, "MLDB.py", f"update_model() {lineno()}")
        return 0, err_str

    # Success!
    msg = f"Model {model}'s {item} was updated to {in_val} "
    write_log(msg, 1, "MLDB.py", f"update_model() {lineno()} ")
    return 1, msg


def get_models(status):
    """
    get_models()

    Input:
    - status
    -- 'G' will return all trained models
    -- 'B' will return all untrained models
    --- Note: untrained models do not exist as objects, but are records in DB
    --- so when the model is generated, the status needs to be changed

    Outputs:
    - 0 of failed, 1 if good
    - list of values in hexidecimal format
    - msg for logging puposes
    """
    l_source = "MLDB.py"
    l_note = "update_models() "

    if status not in model_status:
        err_str = "Invalid status: " + status
        write_log(err_str, 2, "MLDB.py", f" get_models() {lineno()}")
        return 0, [], err_str

    out_list = []
    try:
        out_list = su.get_models_by_status(status)
    except Exception as e:
        err_msg = f"Failure to retrieve models due to: {format(e)}"
        write_log(err_msg, 3, l_source, l_note + lineno())
        return 0, [], err_msg

    out_set = []
    for item in out_list:
        out_set.append(item.modelname)
    return 1, out_set, ""


# =============
# Frequency functions
# =============


def new_multi_frequency(model, ldict):
    if isinstance(model, str):
        try:
            model = su.get_model_id(model)
        except Exception as e:
            err_msg = f"Could not get id for {model}: {format(e)}"
            write_log(err_msg, 2, "MLDB.py", "new_multi_frequency()")
            return 0, err_msg

    if len(ldict) == 0:
        err_msg = "No data provided"
        write_log(err_msg, 2, "MLDB.py", "new_multi_frequency()")
        return 0, err_msg

    # Test for existing data and remove it if needed
    if su.test_frequency_model(model):
        su.delete_frequency_records(model)

    total = 0
    for item in ldict:
        for key, value in item.items():
            total = total + value

    print(f"final total: {total}")
    if 0.99 <= total > 1.01:
        err_msg = "Frequencies do not add up to ~ 1.0"
        write_log(err_msg, 2, "MLDB.py", "new_multi_frequency()")
        return 0, err_msg

    for item in ldict:
        for key, value in item.items():
            print(f"{key}, {value}")
            ret, msg = new_frequency(model, key, value)
            if ret == 0:
                return 0, msg

    return 1, f"{len(ldict)} items processed"


def new_frequency(in_model, computer, frequency):
    if isinstance(in_model, str):
        in_model = su.get_model_id(in_model)
    if not isinstance(in_model, int):
        err_msg = f"Invalid format for {in_model}"
        write_log(err_msg, 2, "MLDB.py", "new_frequency()")
        return 0, err_msg

    if not is_valid_name(computer, 32):
        err_msg = f"Invalid computer name: {computer}"
        write_log(err_msg, 2, "MLDB.py", "new_frequency()")
        return 0, err_msg

    if not (0.0 < frequency <= 1.0):
        err_msg = f"Invalid frequency input {frequency}"
        write_log(err_msg, 2, "MLDB.py", "new_frequency()")
        return 0, err_msg

    try:
        f_id = su.insert_frequency(in_model, computer, frequency)
        write_log("New Frequency record added", 1, "MLDB.py", "new_frequency()")
        return 1, f_id
    except Exception as e:
        err_msg = "Problem inserting new frequency: "
        err_msg = err_msg + format(e)
        write_log(err_msg, 2, "MLDB.py", "new_frequency()")
        return 0, err_msg


def get_frequencies(in_model):
    if isinstance(in_model, str):
        in_model = su.get_model_id(in_model)
    if not isinstance(in_model, int):
        err_msg = f"{in_model} is not a valid input"
        write_log(err_msg, 2, "MLDB.py", "get_frequencies()")
        return 0, err_msg

    try:
        ret_set = su.get_frequency(in_model)
        ldict = []
        for item in ret_set:
            c_name = item[0]
            freq = item[1]
            in_dict = {c_name: freq}
            ldict.append(in_dict)

        return 1, ldict
    except Exception as e:
        err_msg = (
            f"Failure to retrieve frequency data for model {in_model} - {format(e)}"
        )
        write_log(err_msg, 2, "MLDB.py", "get_frequencies()")
        return 0, err_msg


# ==========================
# Working Days
# ==========================


def valid_time(in_val):
    if isinstance(in_val, int):
        return 0 <= in_val <= 23
    return False


def days_good_to_go(wdays):
    tot_time = 0.0
    for value in wdays.values():
        if value is not None:
            tot_time = tot_time + value

    return 0.9 < tot_time <= 1.0


def new_working_days(in_user, wdays):
    if isinstance(in_user, str):
        in_user = su.get_user_id(in_user)

    if not (is_valid_id("users", in_user)):
        return 0, "Unidentified user"

    if days_good_to_go(wdays) is False:
        err_msg = "Day values do not total ~ 1.0"
        write_log(err_msg, 2, "MLDB.py", "new_working_days()")
        return 0, err_msg

    try:
        if not su.test_working_days_user(in_user):
            su.insert_w_days(
                in_user,
                wdays["monday"],
                wdays["tuesday"],
                wdays["wednesday"],
                wdays["thursday"],
                wdays["friday"],
                wdays["saturday"],
                wdays["sunday"],
            )

            ret_msg = f"Workings days for user {in_user} inserted successfully"
            write_log(ret_msg, 1, "MLDB.py", "new_working_days()")
            return 1, ret_msg
        else:
            su.update_w_days(
                in_user,
                wdays["monday"],
                wdays["tuesday"],
                wdays["wednesday"],
                wdays["thursday"],
                wdays["friday"],
                wdays["saturday"],
                wdays["sunday"],
            )
            ret_msg = f"Workings days for user {in_user} updated successfully"
            write_log(ret_msg, 1, "MLDB.py", "new_working_days()")
            return 1, ret_msg
    except Exception as e:
        err_msg = f"Problem insertiong working days: {format(e)}"
        write_log(err_msg, 2, "MLDB.py", "new_working_days()")
        return 0, err_msg


def new_work_hours(in_user, whours):
    if isinstance(in_user, str):
        in_user = su.get_user_id(in_user)

    if not is_valid_id("users", in_user):
        err_msg = "Invalid user input"
        write_log(err_msg, 2, "MLDB.py", "new_work_hours()")
        return 0, err_msg

    for key, val in whours.items():
        if not valid_time(val):
            err_msg = f"{key} does not have a valid time value"
            write_log(err_msg, 2, "MLDB.py", "new_work_hours()")
            return 0, err_msg

    try:
        if not su.test_work_hours_user(in_user):
            print("inserting")
            su.insert_work_hours(
                in_user,
                whours["mon_s"],
                whours["mon_e"],
                whours["tues_s"],
                whours["tues_e"],
                whours["wed_s"],
                whours["wed_e"],
                whours["thur_s"],
                whours["thur_e"],
                whours["fri_s"],
                whours["fri_e"],
                whours["sat_s"],
                whours["sat_e"],
                whours["sun_s"],
                whours["sun_e"],
            )
            ret_msg = f"Work hours entered for user {in_user}"
            write_log(ret_msg, 1, "MLDB.py", "new_work_hours()")
            return 1, ret_msg
        else:
            print("updating")
            su.update_work_hours(
                in_user,
                whours["mon_s"],
                whours["mon_e"],
                whours["tues_s"],
                whours["tues_e"],
                whours["wed_s"],
                whours["wed_e"],
                whours["thur_s"],
                whours["thur_e"],
                whours["fri_s"],
                whours["fri_e"],
                whours["sat_s"],
                whours["sat_e"],
                whours["sun_s"],
                whours["sun_e"],
            )
    except Exception as e:
        err_msg = f"Problem inserting work hours for user {in_user}\n{format(e)}"
        write_log(err_msg, 2, "MLDB.py", "new_work_hours()")


def get_work_data(in_user):
    if isinstance(in_user, str):
        _, in_user = su.get_user_id(in_user)
    if not su.test_user_id(in_user):
        err_msg = f"User id: {in_user} could not be found in DB"
        write_log(err_msg, 3, "MLDB.py", "get_work_data()")
        return 0, err_msg

    try:
        wdays = su.get_working_days(in_user)
        whours = su.get_work_hours(in_user)
    except Exception as e:
        err_msg = f"Problem retrieving work data: {format(e)}"
        write_log(err_msg, 3, "MLDB.py", "get_work_data()")

    if len(wdays) == 0 or len(whours) == 0:
        return 0, {}

    wd_dict = {
        "monday": wdays[0],
        "mon_s": whours[0],
        "mon_e": whours[1],
        "tuesday": wdays[1],
        "tues_s": whours[2],
        "tues_e": whours[3],
        "wednesday": wdays[2],
        "wed_s": whours[4],
        "wed_e": whours[5],
        "thursday": wdays[3],
        "thur_s": whours[6],
        "thur_e": whours[7],
        "friday": wdays[4],
        "fri_s": whours[8],
        "fri_e": whours[9],
        "saturday": wdays[5],
        "sat_s": whours[10],
        "sat_e": whours[11],
        "sunday": wdays[6],
        "sun_s": whours[12],
        "sun_e": whours[13],
    }

    return 1, wd_dict


""" End of Module MLDB """
