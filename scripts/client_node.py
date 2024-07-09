# -*- coding: utf-8 -*-
from cair_libraries.client_personalization_server import PersonalizationServer
from cair_libraries.dialogue_statistics import DialogueStatistics
from cair_libraries.dialogue_state import DialogueState
from cair_libraries.dialogue_turn import DialogueTurn
from cair_libraries.client_alterego_utils import AlteregoClientUtils
from playsound import playsound
from gtts import gTTS
from openai import OpenAI
from pydub import AudioSegment
from pydub.playback import play
from mutagen.mp3 import MP3
from dotenv import load_dotenv, find_dotenv
from cairclient_alterego_vision.srv import GestureService
import xml.etree.ElementTree as ET
import pygame.camera
import threading
import requests
import argparse
import socket
import base64
import random
import zlib
import json
import time
import xml
import sys
import re
import os
import rospy
import rospkg
from datetime import datetime

rp = rospkg.RosPack()
package_path = rp.get_path('cairclient_alterego_vision')
folder_path = package_path + "/common"

dense_cap = False
log_data = False

lab_server = "130.251.13.192"

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    # This IP doesn't need to be reachable, it's used only to get the local IP
    s.connect(('8.8.8.8', 1))
    local = s.getsockname()[0]
except ConnectionError:
    local = '127.0.0.1'
finally:
    s.close()

# Set the location of the server
server_ip = lab_server
audio_recorder_ip = local
registration_ip = local
language = "it-IT"
MAX_HISTORY_TURNS = 6
# Silence time in seconds after which we consider the conversation not ongoing
SILENCE_THRESHOLD = 300

server_port = "12345"
img_port = "12348"

BASE_CAIR_hub = "http://" + server_ip + ":" + server_port + "/CAIR_hub"
img_url = "http://" + server_ip + ":" + img_port + "/CAIR_dense_captioning"

# If dense captioning is set to false the dense_cap_result will be sent empty to the server and the
# visual information will not be used by gpt-4
dense_cap_result = []


def mp3_duration(path):
    try:
        audio = MP3(path)
        length = audio.info.length
        return length
    except:
        return None


class CAIRclient:
    def __init__(self):
        # Instances of the classes of the other files in the libs folder containing functions needed here
        self.utils = AlteregoClientUtils(server_port, server_ip, registration_ip)
        self.isAlive = True
        self.exit_keywords = ["stop talking", "esci dallapp", "esci dallapplicazione", "quit the application"]
        self.repeat_keywords = ["repeat", "can you repeat", "say it again", "puoi ripetere", "ripeti", "non ho capito"]
        self.dialogue_state_file_path = os.path.join(folder_path, "dialogue_state.json")
        self.speakers_info_file_path = os.path.join(folder_path, "speakers_info.json")
        self.dialogue_statistics_file_path = os.path.join(folder_path, "dialogue_statistics.json")
        self.nuance_vectors_file_path = os.path.join(folder_path, "nuance_vectors.json")
        self.camera_image_path = os.path.join(folder_path, "camera_image.jpg")
        self.audio_file_path = os.path.join(folder_path, "audio.mp3")

        # To store the previous sentence said by the robot
        self.previous_sentence = ""
        self.dialogue_sentence = []
        self.dialogue_state = {}
        self.dialogue_statistics = {}
        self.speakers_info = {}
        self.nuance_vectors = {}
        self.conversation_history = []
        self.plan_sentence = ""
        self.plan = ""
        # This variable tells if the user want the robot to repeat a sentence
        self.repeat = False
        self.connection_lost = False

        # Read the sentences in files
        self.sentences = {}
        languages = ["it-IT", "en-US", "fr-FR", "es-ES"]
        for lan in languages:
            with open(os.path.join(folder_path, "trigger_sentences_" + lan + ".txt")) as f:
                self.sentences[lan] = [line.rstrip() for line in f]

        self.dense_cap = dense_cap
        self.dense_cap_result = dense_cap_result
        self.microphone_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.log_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.log_data = log_data
        self.due_intervention = {"type": None, "exclusive": False}
        self.personalization_data = PersonalizationServer()
        self.offset = 0.0

    def gesture_service_client(self, filename, audio_duration, offset):
        # Blocking call that waits for server
        rospy.wait_for_service('gesture_service')
        try:
            gesture_service = rospy.ServiceProxy('gesture_service', GestureService)
            response = gesture_service(filename, audio_duration, offset)
            self.offset = response.offset
            rospy.loginfo("**NEW OFFSET: %f", self.offset)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)

    def send_log_data(self, to_log):
        if self.log_data:
            self.log_socket.send(to_log.encode("utf-8"))

    @staticmethod
    def check_socket_connection(received_string):
        if received_string == "":
            if language == "it-IT":
                to_say = "Mi dispiace, c'è stato qualche problema con la connessione al microfono esterno."
            else:
                to_say = "I'm sorry, there was a problem with the connection to the external microphone."
            print("R:", to_say)
            # tts = gTTS(to_say, lang=language)
            # tts.save(self.audio_file_path)
            # playsound(self.audio_file_path)
            # os.system("afplay audio.mp3")
            # stream_and_play(to_say)
            sys.exit(1)

    @staticmethod
    def print_used_nuances(dialogue_nuances):
        print("________USED NUANCES________")
        for key, value in dialogue_nuances["flags"].items():
            flag_index = value.index(1)
            if flag_index == len(value) - 1:
                print(key + ":", dialogue_nuances["values"][key])
            else:
                print(key + ":", dialogue_nuances["values"][key][flag_index])
        print("____________________________")

    def acquire_image(self, cam):
        print("Image thread: start acquiring camera input")
        cam.start()
        # Grab the first frame
        img = cam.get_image()
        WIDTH = img.get_width()
        HEIGHT = img.get_height()
        # display = pygame.display.set_mode((WIDTH, HEIGHT), 0)
        screen = pygame.surface.Surface((WIDTH, HEIGHT))

        while True:
            # Take the timestamp at the beginning of the loop to process vision information
            now = datetime.now()
            date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
            # Capture an image
            image_capture_start_time = time.time()
            screen = cam.get_image(screen)
            # Save the image
            pygame.image.save(screen, "camera_image.jpg")
            with open("camera_image.jpg", "rb") as img_file:
                # Encode your image with base64 and convert it to string
                img_encoded = base64.b64encode(img_file.read()).decode('utf-8')
            image_capture_end_time = time.time()
            image_byte_size = (len(img_encoded)*3)/4

            # Create a dictionary with the encoded image
            data = {"frame": img_encoded}
            # Send the image to the server using a POST request
            try:
                # Log the time needed to perform the request and get the response
                request_start_time = time.time()
                response = requests.post(img_url, json=data)
                self.dense_cap_result = response.json()["result"]
                request_end_time = time.time()
                to_log = "v#timestamp:" + date_time_str + "\n" \
                         "v#image_capture_time:" + str(image_capture_end_time - image_capture_start_time) + "\n" \
                         "v#compressed_image_size_bytes:" + str(image_byte_size) + "\n" \
                         "v#densecap_request_response_time:" + str(request_end_time - request_start_time) + "\n" \
                         "v#********************\n"
                self.send_log_data(to_log)
            except ConnectionError:
                print("** The Dense Captioning service is not available")
                exit(1)

    def say_filler_sentence(self):
        now = datetime.now()
        date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        to_log = "d#timestamp:" + date_time_str + "\n"
        self.send_log_data(to_log)
        filler_sentence = random.choice(self.sentences[language])
        print("FILLER:", filler_sentence)
        # Log the time taken to say the random sentence
        start_time = time.time()
        tts = gTTS(filler_sentence, lang=language.split('-')[0])
        tts.save(self.audio_file_path)
        playsound(self.audio_file_path)
        end_time = time.time()
        #   os.system("afplay audio.mp3")
        # stream_and_play(filler_sentence)
        to_log = "d#ack_sentence_speaking_time:" + str(end_time - start_time) + "\n"
        self.send_log_data(to_log)

    def connect_to_audio_and_log_services(self):
        # Try connecting to the socket that records the audio
        print("Trying to connect to the audio recorder socket.")
        try:
            self.microphone_socket.connect((audio_recorder_ip, 9090))
            # the service to log data is on 9092 as on 9091 there is the registration service
            if self.log_data:
                self.log_socket.connect((audio_recorder_ip, 9092))
        except ConnectionError:
            if language == "it-IT":
                to_say = "Mi dispiace, non riesco a connettermi al microfono o al servizio di log."
            else:
                to_say = "I'm sorry, I can't connect to the microphone or the log service"
            print("R:", to_say)
            tts = gTTS(to_say, lang=language.split("-")[0])
            tts.save(self.audio_file_path)
            playsound(self.audio_file_path)
            os.remove(self.audio_file_path)
            sys.exit(1)

    def initialize_user_session(self):
        # If it's the first time using the system, call the function that acquires the first state
        if not os.path.isfile(self.speakers_info_file_path):
            print("First user!")
            # This function creates the speakers_info and the speakers_sequence_stats files and initializes them
            # with the info of a generic user
            welcome_sentence_str = self.utils.acquire_initial_state(language)
            # Retrieve starting nuance vectors
            with open(self.nuance_vectors_file_path) as f:
                self.nuance_vectors = json.load(f)
            welcome_str = welcome_sentence_str
        else:
            print("Users are already present in the info file")
            if language == "it-IT":
                welcome_back_msg = "È bello rivedervi! Di cosa vorreste parlare?"
            else:
                welcome_back_msg = "Welcome back! I missed you. What would you like to talk about?"
            welcome_str = welcome_back_msg

        print("R:", welcome_str)
        tts = gTTS(welcome_str, lang=language.split('-')[0])
        tts.save(self.audio_file_path)
        duration = mp3_duration(self.audio_file_path)
        filename = "talk" + str(random.randint(0, 9)) + ".bag"
        welcome_thread = threading.Thread(None, self.gesture_service_client, args=(filename, duration, 0,))
        welcome_thread.start()
        time.sleep(1)
        playsound(self.audio_file_path)
        # stream_and_play(welcome_str)
        self.previous_sentence = welcome_str

    def load_conversation_state(self):
        # Retrieve the state of the conversation and save it in a dictionary
        with open(self.dialogue_state_file_path) as f:
            self.dialogue_state = DialogueState(d=json.load(f))

        # If it is the first time, fill the nuance vectors from the file
        if len(self.nuance_vectors) != 0:
            self.dialogue_state.dialogue_nuances = self.nuance_vectors
        # Store the welcome or welcome back string in the assistant field
        self.dialogue_state.conversation_history.append({"role": "assistant", "content": self.previous_sentence})

        # Retrieve the info of the users and store them in a dictionary
        with open(self.speakers_info_file_path) as f:
            self.speakers_info = json.load(f)

        # Retrieve dialogue statistics file
        with open(self.dialogue_statistics_file_path) as f:
            self.dialogue_statistics = DialogueStatistics(d=json.load(f))

        self.dialogue_state.prev_dialogue_sentence = [["s", self.previous_sentence]]

    def start_dense_captioning_in_thread(self):
        # Initialize camera
        pygame.camera.init()
        cam_list = pygame.camera.list_cameras()
        cam = pygame.camera.Camera(cam_list[0])
        t1 = threading.Thread(None, self.acquire_image, args=(cam,))
        t1.start()

    def hub_request(self, data):
        encoded_data = json.dumps(data).encode('utf-8')
        compressed_data = zlib.compress(encoded_data)
        start_time = time.time()
        hub_response = requests.get(BASE_CAIR_hub, data=compressed_data, verify=False)
        end_time = time.time()
        # If the Hub cannot contact the dialogue service, the response will be empty
        if hub_response:
            error = hub_response.json().get("error", "")
            if error != "":
                print(error)
                exit(1)
            # Overwrite the array containing the states of the profiles with those contained in the Hub response
            # The speakers info are not sent to the Hub.
            self.dialogue_state = DialogueState(d=hub_response.json()['dialogue_state'])
            # Store the updated dialogue state in the file
            with open(self.dialogue_state_file_path, 'w') as f:
                json.dump(self.dialogue_state.to_dict(), f, ensure_ascii=False, indent=4)
            self.dialogue_sentence = hub_response.json()['dialogue_sentence']

            if data["req_type"] == "reply":
                to_log = "d#first_request_response_time:" + str(end_time - start_time) + "\n"
                self.send_log_data(to_log)
                # self.print_used_nuances(self.dialogue_state.dialogue_nuances)
                self.dialogue_statistics = DialogueStatistics(d=hub_response.json()["dialogue_statistics"])
                # The hub updates the average topic distance matrix, hence it should be written on the file
                with open(self.dialogue_statistics_file_path, 'w') as f:
                    json.dump(self.dialogue_statistics.to_dict(), f, ensure_ascii=False, indent=4)
                self.plan_sentence = hub_response.json()['plan_sentence']
                self.plan = hub_response.json()['plan']
                self.due_intervention = hub_response.json()["due_intervention"]
                print("Due intervention returned by Hub:", self.due_intervention)
            else:
                to_log = "d#second_request_response_time:" + str(end_time-start_time) + "\n"
                self.send_log_data(to_log)
        else:
            print("No response received from the Hub!")
            self.connection_lost = True

    def start_dialogue(self):
        global language
        ongoing_conversation = True

        print("Starting personalization server in thread")
        self.personalization_data.start_server_in_thread()

        self.connect_to_audio_and_log_services()

        self.initialize_user_session()
        self.load_conversation_state()

        prev_turn_last_speaker = ""
        prev_speaker_topic = ""

        # If dense captioning should be used, start the thread to update visual information
        if self.dense_cap:
            self.start_dense_captioning_in_thread()

        last_active_speaker_time = time.time()

        while self.isAlive:
            self.offset = 0.0
            filename = "talk" + str(random.randint(0, 9)) + ".bag"
            print("** Listening **")
            # Check if the conversation is ongoing
            if time.time() - last_active_speaker_time > SILENCE_THRESHOLD:
                print("Silence threshold exceeded - setting ongoing_conversation to False")
                ongoing_conversation = False

            # Check if there is a due scheduled intervention
            due_intervention = self.personalization_data.get_due_intervention()

            # If there is no scheduled intervention, reset the type of the class variable to None, meaning that the
            # xml_string will contain a user sentence and not a intervention sentence
            if due_intervention is None:
                self.due_intervention["type"] = None
            else:
                self.due_intervention = due_intervention

            print("---- DUE INTERVENTION:", self.due_intervention, "----")
            if self.due_intervention["type"] is None:
                print("** Listening **")
                if os.path.exists(self.audio_file_path):
                    os.remove(self.audio_file_path)
                # Tell the audio recorder that the client is ready to receive the user reply
                self.microphone_socket.send(self.dialogue_state.sentence_type.encode("utf-8"))
                # The first string received can be an ack that the user has finished talking (after 2s) or a timeout
                received_str = self.microphone_socket.recv(1024).decode('utf-8')
                # If the received string is empty it means that the connection is broken
                self.check_socket_connection(received_str)
                # If the received string is timeout it means that the user remained in silence for some time
                if received_str == "timeout":
                    print("** Received timeout from microphone **", "blue")
                    continue
                self.microphone_socket.send("ack".encode("utf-8"))
                if time.time() - last_active_speaker_time <= SILENCE_THRESHOLD:
                    ongoing_conversation = True
                last_active_speaker_time = time.time()

            if self.due_intervention["type"] is None:
                # The second string received should be the xml client sentence
                xml_string = self.microphone_socket.recv(1024).decode('utf-8')
                self.check_socket_connection(xml_string)

                # Do not proceed until the xml string is complete and all tags are closed
                proceed = False
                while not proceed:
                    try:
                        ET.ElementTree(ET.fromstring(xml_string))
                        proceed = True
                    except xml.etree.ElementTree.ParseError:
                        # If the xml is not complete, read again from the socket
                        print("The XML is not complete.")
                        xml_string = xml_string + self.microphone_socket.recv(1024).decode('utf-8')
            else:
                # If there is a due intervention, create the xml string with the sentence of the intervention
                xml_string = ('<response><profile_id value="' + self.dialogue_statistics.mapping_index_speaker[int(0)] +
                              '">' + self.due_intervention["sentence"] + '<language>' + language +
                              '</language><speaking_time>1</speaking_time></profile_id></response>')
                # Remove the sentence field from the dictionary as now it is stored in the xml string
                self.due_intervention.pop("sentence")

            # Create a dialogue turn object starting from the xml
            dialogue_turn = DialogueTurn(xml_string)

            if self.due_intervention["type"] is None:
                # Update the dialogue statistics only if the required minimum number of users is registered
                if len(self.dialogue_statistics.mapping_index_speaker) > 1:
                    self.dialogue_statistics.update_statistics(dialogue_turn, prev_turn_last_speaker)

                    # Update content of the speaker stats file after having updated them after someone talked
                    with open(self.dialogue_statistics_file_path, 'w') as cl_state:
                        json.dump(self.dialogue_statistics.to_dict(), cl_state, ensure_ascii=False, indent=4)

            # Parse the xml string and extract the first sentence and the first speaker
            tree = ET.ElementTree(ET.fromstring(xml_string))
            speaker_id = tree.findall('profile_id')[0]
            language = speaker_id.find("language").text
            sentence = tree.findall('profile_id')[0].text.strip('.,!?')
            print("USER:", sentence)

            # Initialize and start the thread that says something to fill the void while waiting
            filler_sentence_thread = threading.Thread(None, self.say_filler_sentence, args=())
            if due_intervention is None:
                filler_sentence_thread.start()

            sentence = sentence.replace(".", "")
            # Reset repeat to false, otherwise it will always repeat the previous sentence
            self.repeat = False

            # Check if the user wants to exit or wants the robot to repeat the previous sentence
            # If the user said one of the "Exit Application keywords"
            if any(exit_sent in sentence for exit_sent in self.exit_keywords) and self.due_intervention["type"] is None:
                self.isAlive = False
                if language == "it-IT":
                    goodbye_msg = "Ok, è stato bello passare del tempo insieme! A presto!"
                else:
                    goodbye_msg = "Ok, it was a pleasure talking with you! Goodbye."
                print("R:", goodbye_msg)
                tts = gTTS(goodbye_msg, lang=language.split('-')[0])
                tts.save(self.audio_file_path)
                duration = mp3_duration(self.audio_file_path)
                goodbye_thread = threading.Thread(None, self.gesture_service_client,
                                                  args=(filename, duration, self.offset,))
                goodbye_thread.start()
                time.sleep(1)
                playsound(self.audio_file_path)
                sys.exit(0)
            # If the user said a Repeat keyword
            elif sentence.lower() in self.repeat_keywords and self.due_intervention["type"] is None:
                # If a previous sentence to repeat exists
                if self.previous_sentence:
                    self.repeat = True
                    if language == "it-IT":
                        repeat_msg = "Certamente. Ho detto: "
                    else:
                        repeat_msg = "Sure! I said: "
                    print("R: " + repeat_msg + self.previous_sentence)
                    tts = gTTS(repeat_msg + self.previous_sentence, lang=language.split('-')[0])
                    tts.save(self.audio_file_path)
                    playsound(self.audio_file_path)
                else:
                    if language == "it-IT":
                        repeat_msg = "Mi dispiace, non ho niente da ripetere."
                    else:
                        repeat_msg = "I'm sorry, I have nothing to repeat."
                    print("R:", repeat_msg)
                    tts = gTTS(repeat_msg, lang=language.split('-')[0])
                    tts.save(self.audio_file_path)
                    playsound(self.audio_file_path)
                    # os.system("afplay audio.mp3")

            # If the user did not ask to exit or to repeat something, send the sentence to the server
            if not self.repeat:
                # Do not add the trigger for action interventions!
                if self.due_intervention["type"] != "action":
                    # Store the user sentence in the conversation history of the dialogue state and pop the first item if needed
                    if len(self.dialogue_state.conversation_history) >= MAX_HISTORY_TURNS:
                        self.dialogue_state.conversation_history.pop(0)
                    self.dialogue_state.conversation_history.append({"role": "user", "content": sentence})

                # Add the parameter ongoing_conversation to the dialogue state
                self.dialogue_state.ongoing_conversation = ongoing_conversation

                # Copy the speakers info in a dictionary that does not contain the names
                # This is needed by OpenAI as it should know the gender.
                speakers_info_no_names = {}
                for speaker_id in self.speakers_info:
                    speakers_info_no_names[speaker_id] = {"gender": self.speakers_info[speaker_id]["gender"],
                                                          "age": self.speakers_info[speaker_id]["age"]}

                # Compose the payload of the message to be sent to the server
                data = {"req_type": "reply",
                        "client_sentence": xml_string, "language": language,
                        "due_intervention": self.due_intervention,
                        "dialogue_state": self.dialogue_state.to_dict(),
                        "dialogue_statistics": self.dialogue_statistics.to_dict(),
                        "speakers_info": speakers_info_no_names,
                        "prev_speaker_info": {"id": prev_turn_last_speaker, "topic": prev_speaker_topic},
                        "dense_cap_result": self.dense_cap_result}

                if self.due_intervention["type"] is None:
                    # Update the info about id and topic of previous speaker to the current one
                    prev_turn_last_speaker = dialogue_turn.turn_pieces[-1].profile_id
                    prev_speaker_topic = self.dialogue_state.topic

                # Create the thread for the first request
                req1_thread = threading.Thread(target=self.hub_request, args=(data,))
                req1_thread.start()
                # Wait for filler sentence thread only if there is no intervention to do
                if self.due_intervention["type"] is None:
                    filler_sentence_thread.join()
                # Wait for the thread to finish
                req1_thread.join()
                if self.connection_lost:
                    print("Connection with Hub lost")
                    exit(1)

                # If there is a plan sentence, it means that something has been matched by the Plan manager service
                if self.plan_sentence:
                    print("PLAN SENTENCE:", self.plan_sentence)
                    self.plan_sentence = self.utils.replace_speaker_name(self.plan_sentence, self.speakers_info)
                    self.plan_sentence = \
                        (self.utils.replace_schwa_in_string(self.plan_sentence, self.speakers_info, speaker_id))
                    tts = gTTS(self.plan_sentence, lang=language.split('-')[0])
                    tts.save(self.audio_file_path)
                    playsound(self.audio_file_path)
                    # os.system("afplay audio.mp3")
                    # stream_and_play(self.plan_sentence)

                # If there is a plan, execute it (if the behavior is installed)
                if self.plan:
                    plan_items = self.plan.split("#")[1:]
                    # For each action in the plan, check which action is it and execute it
                    for item in plan_items:
                        action = re.findall("action=(\w+)", item)[0]
                        if action == "registration":
                            # The function that manages the registration, updates the files and returns the updated
                            # dictionaries, so that we don't have to read from the files at each turn.
                            self.speakers_info, self.dialogue_statistics = self.utils.registration_procedure(language)
                        else:
                            if language == "it-IT":
                                to_say = action
                            else:
                                to_say = "I'm sorry, I'm still not able to perform actions"
                            print("ACTION: perform action '" + to_say + "'")
                            # tts = gTTS(to_say, lang=language.split('-')[0])
                            # tts.save(self.audio_file_path)
                            # playsound(self.audio_file_path)
                            # os.system("afplay audio.mp3")

                # Make a copy of the dialogue sentence before it is modified by the second request
                # When there is a scheduled intervention, this will be a list of one tuple ['r', ''].
                dialogue_sentence1_str = ""
                dialogue_sentence1_history = ""
                if self.dialogue_sentence[0][1] != "":
                    dialogue_sentence1 = self.dialogue_sentence
                    dialogue_sentence1_history = self.utils.process_sentence(dialogue_sentence1, self.speakers_info)
                    # Store the assistant sentence in the conversation history of the dialogue state
                    if len(self.dialogue_state.conversation_history) >= MAX_HISTORY_TURNS:
                        self.dialogue_state.conversation_history.pop(0)
                    self.dialogue_state.conversation_history.append(
                        {"role": "assistant", "content": dialogue_sentence1_history})
                    dialogue_sentence1_str = self.utils.replace_speaker_name(dialogue_sentence1_history, self.speakers_info)

                # Update the data content for the second request - update also the dialogue state!
                data["req_type"] = "continuation"
                # print(self.dialogue_state.to_dict())
                data["dialogue_state"] = self.dialogue_state.to_dict()
                data["dense_cap_result"] = self.dense_cap_result
                # Empty the field of the due intervention as it has already been processed by the first request
                # Create a thread that performs another request to get the continuation of the dialogue
                req2_thread = threading.Thread(target=self.hub_request, args=(data,))
                req2_thread.start()

                start_time = time.time()
                dialogue1_thread = threading.Thread(None, self.gesture_service_client,
                                                    args=(filename, duration, self.offset,))
                # Avoid speaking first part of dialogue sentence if empty (when there is an action intervention)
                if dialogue_sentence1_str != "":
                    print("REPLY:", dialogue_sentence1_str)
                    tts = gTTS(dialogue_sentence1_str, lang=language.split('-')[0])
                    tts.save(self.audio_file_path)
                    duration = mp3_duration(self.audio_file_path)
                    # Wait for the previous gesture thread to finish
                    filler_sentence_thread.join()
                    dialogue1_thread.start()
                    time.sleep(1)
                    playsound(self.audio_file_path)
                    end_time = time.time()
                    # os.system("afplay audio.mp3")
                    # stream_and_play(dialogue_sentence1_str)
                    to_log = "d#first_response_speaking_time:" + str(end_time - start_time) + "\n"
                    self.send_log_data(to_log)

                if req2_thread.is_alive():
                    req2_thread.join()
                if self.connection_lost:
                    print("Connection with Hub lost")
                    exit(1)

                dialogue2_thread = threading.Thread(None, self.gesture_service_client,
                                                    args=(filename, duration, self.offset,))
                if self.dialogue_sentence[1][1] != "":
                    dialogue_sentence2 = self.dialogue_sentence[1:]
                    dialogue_sentence2_history = self.utils.process_sentence(dialogue_sentence2, self.speakers_info)

                    dialogue_sentence2_str = self.utils.replace_speaker_name(dialogue_sentence2_history,
                                                                             self.speakers_info)
                    print("CONTINUATION:", dialogue_sentence2_str)
                    start_time = time.time()
                    tts = gTTS(dialogue_sentence2_str, lang=language.split('-')[0])
                    tts.save(self.audio_file_path)
                    duration = mp3_duration(self.audio_file_path)
                    # Wait for the previous gesture thread to finish
                    dialogue1_thread.join()
                    dialogue2_thread.start()
                    time.sleep(1)
                    playsound(self.audio_file_path)
                    end_time = time.time()
                    # os.system("afplay audio.mp3")
                    # stream_and_play(dialogue_sentence2_str)
                    to_log = "d#second_sentence_speaking_time:" + str(end_time - start_time) + "\n" \
                             "d#********************\n"
                    self.send_log_data(to_log)

                    # Replace the last assistant reply with the complete one
                    self.dialogue_state.conversation_history.pop()
                    self.dialogue_state.conversation_history.append(
                        {"role": "assistant", "content": dialogue_sentence1_history + " " + dialogue_sentence2_history})
                    self.dialogue_state.prev_dialogue_sentence = self.dialogue_sentence
                # If there is no continuation response it means there was an action intervention
                else:
                    # If there was an ongoing conversation, repeat the previous continuation sentence
                    if ongoing_conversation:
                        map_language_sentence = {"it-IT": "Dicevo...",
                                                 "en-US": "I was saying...",
                                                 "fr-FR": "Je disais...",
                                                 "es-ES": "Decía..."}
                        print(self.dialogue_state.prev_dialogue_sentence)
                        repeat_continuation = map_language_sentence[language] + self.dialogue_state.prev_dialogue_sentence[-1][1]
                        print("REPEAT CONTINUATION:", repeat_continuation)
                        tts = gTTS(repeat_continuation, lang=language.split('-')[0])
                        tts.save("audio.mp3")
                        playsound("audio.mp3")


if __name__ == '__main__':
    # Define the program description
    text = 'This is the client for CAIR.'
    # Initiate the parser with a description
    parser = argparse.ArgumentParser(description=text)
    # Add long and short argument
    parser.add_argument("--language", "-l", help="set the language of the client to it-IT or en-US")

    # Read arguments from the command line
    args = parser.parse_args()
    if not args.language:
        print("No language provided. The default Italian language will be used.")
        language = "it-IT"
    else:
        language = args.language
        print("The language has been set to", language)

    rospy.init_node('gesture_service_client', anonymous=False)
    cair_client = CAIRclient()
    cair_client.start_dialogue()
