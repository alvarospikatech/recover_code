import sys

sys.path.append("src\VRCARDIO-Testing\database_calls")
from procesing_data.launch_api_calls import *
from procesing_data.launch_session_processor import *

sys.path.append("src/VRCARDIO-Testing/vr_cardio_modules_tests/_0_PREPROCESING_TESTS")
import main_preprocesing


def update_all():
    download_sesions()


def update_and_process_all():
    download_sesions()
    process_all_cases()


def create_excells():
    create_basic_excell()
    create_complex_excell()


def process_one_session(user_id, session_id):
    process_one_case(user_id, session_id)


def download_necesary_data():
    update_and_process_all()
    create_excells()


if __name__ == "__main__":

    download_sesions()
    download_all_cases_signals()
    main_preprocesing.calculate_for_all_cases()
    #update_and_process_all()
    #process_all_cases()
    #process_one_session("6fc3dac8-14c6-4a64-8a99-74ad9c10392e", "0974d762-eca6-471c-9e83-3582747332e0")

    #create_excells()
