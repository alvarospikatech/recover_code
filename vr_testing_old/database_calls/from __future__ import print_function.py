"""
Author: SpikaTech
Date: 24/10/2023
Description:
"""

from __future__ import print_function

import time
from pprint import pprint

import swagger_client
from swagger_client.rest import ApiException

# Configure HTTP basic authorization: Basic
configuration = swagger_client.Configuration()


# create an instance of the API class
api_instance = swagger_client.TokenApi(swagger_client.ApiClient(configuration))

api_response4 = api_instance.vr_cardio_sessions_read("fa9be3b7-20ff-4430-be91-ce9f06c6d0a5")

a = api_response4.to_dict()

api_response = api_instance.token_create({"username": "admin", "password": "SpikaTechSL2022"})


api_instance2 = swagger_client.TokenApi(swagger_client.ApiClient())
api_response2 = api_instance2.token_refresh_create({"refresh": api_response.refresh})
pprint(api_response2)


configuration2 = swagger_client.Configuration()
configuration2.api_key = {"Authorization": api_response2.access, "Refresh": api_response.refresh}
# dict to store API prefix (e.g. Bearer)
configuration2.api_key_prefix = {"Authorization": "Bearer "}


api_instance = swagger_client.VrCardioApi(swagger_client.ApiClient(configuration2))
api_response4 = api_instance.vr_cardio_session_list()
pprint(api_response4)
sessionid = "7a97066d-23b7-48bc-9281-4a24448d9f4e"
