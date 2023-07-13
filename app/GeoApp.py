import math
import os
import pandas as pd
import numpy as np
import spacy
from geopy.geocoders import Nominatim
from geopy import distance
import pgeocode
import ssl

import folium
from folium.plugins import HeatMap, MarkerCluster, MeasureControl
from folium.features import DivIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox, QHBoxLayout, QComboBox
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, Qt
from PyQt5.QtGui import QIcon


class GeoApp(QMainWindow):

    def __init__(self):
        '''
        Loading backend database.
        Initialization of GUI.
        '''
        super().__init__()

        # TODO: Edit path to database and model accordingly
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.locations_file = os.path.join(self.current_dir, "data", "locations.pkl")
        self.model_path = os.path.join(self.current_dir, "models", "model-best")
        self.location_data = None
        self.nlp = None

        # Initialize GUI
        self.setWindowTitle("GeoApp")
        self.setWindowIcon(QIcon("misc/map_icon.png"))
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.layout = QVBoxLayout()

        self.label_address = QLabel("Enter the address:")
        self.label_address.setMaximumHeight(20)
        self.input_address = QLineEdit()
        self.input_address.setMaximumHeight(30)
        self.input_address.setPlaceholderText("E.g. 123 ABC Road Singapore 987123")

        self.label_proximity = QLabel("Enter the proximity threshold (in km):")
        self.label_proximity.setMaximumHeight(20)
        self.input_proximity = QLineEdit()
        self.input_proximity.setMaximumHeight(30)
        self.input_proximity.setPlaceholderText("E.g. 2")
        
        self.layout.addWidget(self.label_address)
        self.layout.addWidget(self.input_address)
        self.layout.addWidget(self.label_proximity)
        self.layout.addWidget(self.input_proximity)
        
        self.map_type_box = QComboBox()
        self.map_type_box.addItem("Heat Density")
        self.map_type_box.addItem("Clusters")
        self.map_type_box.addItem("Proximity")
        self.map_type_box.setFixedWidth(300)

        combo_layout = QHBoxLayout()
        combo_layout.addStretch()
        combo_layout.addWidget(self.map_type_box)
        combo_layout.addStretch()

        self.layout.addLayout(combo_layout)

        self.button_ok = QPushButton("Enter")
        self.button_ok.setFixedSize(300, 30)
        self.button_ok.clicked.connect(self.process_user_input)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.button_ok)
        button_layout.addStretch()

        self.layout.addLayout(button_layout)

        self.web_view = QWebEngineView()

        self.layout.addWidget(self.web_view)
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

        # Display GUI in full screen
        self.showMaximized()


    def load_location_data(self):
        """
        Load the location data from the path.
        """
        self.location_data = pd.read_pickle(self.locations_file)[['address', 'postal_code', 'latitude', 'longitude']]


    def load_spacy_model(self):
        '''
        Load the trained spacy model from the path.
        '''
        self.nlp = spacy.load(self.model_path)


    def process_user_input(self):
        '''
        Process user input for map type, source address and proximity threshold.
        '''
        input_address = self.input_address.text()
        proximity_threshold = self.input_proximity.text()
        map_type = self.map_type_box.currentText()

        # Check validity of user input
        if self.check_user_input(input_address, proximity_threshold):
            proximity_threshold = float(proximity_threshold)

            self.load_location_data()
            self.load_spacy_model()

            postal_code = self.extract_postal_code(input_address)
            user_location = self.address_to_lat_long(postal_code, 'geopy')

            if user_location is not None:
                user_latitude, user_longitude = user_location.latitude, user_location.longitude
                location_in_sg = self.check_location_in_sg(user_latitude, user_longitude)

                if location_in_sg == False:
                    # Retry geocoding using pgeocode
                    user_location = self.address_to_lat_long(postal_code, 'pgeocode')
                    user_latitude, user_longitude = user_location.latitude, user_location.longitude
                    location_in_sg = self.check_location_in_sg(user_latitude, user_longitude)

                if location_in_sg == True:
                    # Filter locations based on proximity threshold
                    filtered_locations = self.filter_locations(
                        self.location_data, user_latitude, user_longitude, proximity_threshold
                    )

                    if not filtered_locations.empty:
                        # Print filtered locations in console
                        self.print_addresses(filtered_locations, proximity_threshold)
                        # Display map with filtered locations
                        self.display_map(
                            filtered_locations, input_address, user_latitude, user_longitude, proximity_threshold, map_type
                        )
                    else:
                        # Display map without any filtered locations
                        self.display_map(
                            None, input_address, user_latitude, user_longitude, proximity_threshold
                        )
                        self.display_error_message("No locations found within the specified proximity.")
                else:
                    self.display_error_message("Unable to retrieve valid coordinates in Singapore")
            else:
                self.display_error_message("Unable to retrieve coordinates for the address")


    def check_user_input(self, input_address, proximity_threshold):
        """
        Check the validity of user input.

        Args:
            input_address (str): The user input address
            proximity_threshold (float): The user input proximity threshold

        Returns:
            bool: True if the user input is valid, False otherwise.
        """
        if not input_address:
            self.display_error_message("Please enter an address.")
            return False

        if not proximity_threshold:
            self.display_error_message("Please enter a proximity threshold.")
            return False

        try:
            proximity_threshold = float(proximity_threshold)
            if proximity_threshold <= 0:
                self.display_error_message("Proximity threshold must be a positive number.")
                return False
        except ValueError:
            self.display_error_message("Proximity threshold must be a valid number.")
            return False

        return True
    
    
    def check_location_in_sg(self, latitude, longitude):
        '''
        Checks if the input latitude and longitude lies within Singapore.
        
        Args:
            latitude (float): The user input latitude.
            longitude (float): The user input longitude.
        
        Returns:
            bool: True if location lies within Singapore, False if otherwise.
        '''
        # latitude [1.15, 1.47]
        # longitude [103.6, 104.1]
        
        min_lat_sg = 1.15
        max_lat_sg = 1.47
        min_lng_sg = 103.6
        max_lng_sg = 104.1
        
        if (min_lat_sg <= latitude <= max_lat_sg) and (min_lng_sg <= longitude <= max_lng_sg):
            return True
        return False  


    def display_error_message(self, message):
        """
        Display an error message.

        Args:
            message (str): The error message to display.
        """
        error_box = QMessageBox()
        error_box.setIcon(QMessageBox.Critical)
        error_box.setWindowTitle("Error")
        error_box.setText(message)
        error_box.exec_()  


    def extract_postal_code(self, input_address):
        """
        Extract the postal code from an input address using the loaded nlp model.

        Args:
            input_address (str): The input address.

        Returns:
            str: The extracted postal code, or None if not found.
        """
        # Process input address using NLP model
        doc = self.nlp(input_address)
        ent_list = [(ent.text, ent.label_) for ent in doc.ents]
        postal_code = None

        # Check if any entity is a postal code and extract it
        for ent_text, ent_label in ent_list:
            if ent_label == "POSTAL_CODE":
                postal_code = ent_text
                break

        return postal_code
    

    def address_to_lat_long(self, address, geo_service):
        """
        Convert an address to latitude and longitude coordinates.

        Args:
            address (str): The address to convert.

        Returns:
            geopy.location.Location: The location object containing latitude and longitude coordinates, 
            or None if the coordinates could not be retrieved.
        """
        # geopy method
        if geo_service=='geopy':
            geolocator = Nominatim(user_agent="Geocoder")
            location = geolocator.geocode(address)
            if location:
                print(f"\n[geopy___] Postal code: {address}, coordinates: ({location.latitude}, {location.longitude})")
                return location
            
        # pgeocode method
        elif geo_service=='pgeocode':   
            ssl._create_default_https_context = ssl._create_unverified_context # workaround to use pgeocode    
            geolocator2 = pgeocode.Nominatim('sg')
            location2 = geolocator2.query_postal_code(address)
            if not location2.empty:
                print(f"\n[pgeocode] Postal code: {address}, coordinates: ({location2.latitude}, {location2.longitude})")
                return location2
            
        # can introduce fallback in the future (alternative APIs or geocoding services) to improve geocoding
        else:
            print(f"\nPostal code: {address}, Latitude and Longitude not found")
            return None
        

    def create_folium_map(self, df, input_address, latitude, longitude, proximity_threshold, map_type):
        """
        Create a Folium map and add the relevant components.

        Args:
            df (pandas.DataFrame): DataFrame containing the addresses of the filtered locations.
            input_address (str): The user input address.
            latitude (float): The latitude of the user input location.
            longitude (float): The longitude of the user input location.
            proximity_threshold (float): The proximity threshold input by user.
            map_type (str): The type of map to be generated.

        Returns:
            folium.Map: The created Folium map with relevant components.
        """
        custom_zoom = self.get_zoom_level(proximity_threshold)
        m = folium.Map(location=[latitude, longitude], 
                       zoom_start=custom_zoom)

        if df is not None:
            if map_type == "Heat Density":
                self.add_heat_density_to_map(m, df)
            elif map_type == "Clusters":
                self.add_clusters_to_map(m, df)
            elif map_type == "Proximity":
                self.add_markers_to_map(m, df, latitude, longitude)
                self.add_polyline_to_map(m, df, latitude, longitude) # draw line from source to location at closest proximity

        self.add_input_marker_to_map(m, input_address, latitude, longitude)
        self.add_proximity_circle_to_map(m, latitude, longitude, proximity_threshold)
        
        return m
    

    def get_marker_colour(self, num_addresses):
        """
        Get the marker color based on the number of addresses at a location.

        Args:
            num_addresses (int): The number of addresses at the location.

        Returns:
            str: The marker color.
        """
        if num_addresses > 1:
            return 'darkblue'
        else:
            return 'blue'


    def add_heat_density_to_map(self, m, df):
        """
        Add heat density for filtered locations to the map.

        Args:
            m (folium.Map): The Folium map to make additions on.
            df (pandas.DataFrame): The filtered locations DataFrame.
        """
        heat_data = df[['latitude', 'longitude']].values
        HeatMap(heat_data,
                       radius=15,
                       blur=10,
                       min_opacity=0.4).add_to(m)
        
    
    def add_clusters_to_map(self, m, df):
        """
        Add clusters and markers for filtered locations to the map.

        Args:
            m (folium.Map): The Folium map to make additions on.
            df (pandas.DataFrame): The filtered locations DataFrame.
        """
        # max_cluster_radius is in pixels, consider adjustments to distance
        marker_cluster = MarkerCluster(max_cluster_radius=150).add_to(m)

        for _, row in df.iterrows():
            lat = row['latitude']
            lon = row['longitude']
            popup = folium.Popup(row['address'],
                                 max_width=250)
            folium.Marker(location=[lat, lon], 
                          popup=popup).add_to(marker_cluster)


    def add_markers_to_map(self, m, df, latitude, longitude):
        """
        Add markers for filtered locations to the map.

        Args:
            m (folium.Map): The Folium map to make additions on.
            df (pandas.DataFrame): The filtered locations DataFrame.
            latitude (float): The latitude of the user location.
            longitude (float): The longitude of the user location.
        """
        address_groups = df.groupby(['latitude', 'longitude'])
        for _, group in address_groups:
            lat, lng = group['latitude'].iloc[0], group['longitude'].iloc[0]
            addresses = group['address'].tolist()
            popup_content = "<br><br>".join(addresses)

            # Calculate the distance from the input location
            distance_from_input = distance.distance((latitude, longitude), (lat, lng)).km
            popup_content += f"<br><br>Distance from input: {distance_from_input:.3f} km"

            marker_colour = self.get_marker_colour(len(addresses))

            folium.Marker(
                location=[lat, lng],
                popup=folium.Popup(popup_content, 
                                   max_width=250),
                icon=folium.Icon(icon='fa-location-dot', 
                                 color=marker_colour)
            ).add_to(m)


    def add_input_marker_to_map(self, m, input_address, latitude, longitude):
        """
        Add a marker for the user input address to the map.

        Args:
            m (folium.Map): The Folium map.
            input_address (str): The user input address.
            latitude (float): The latitude of the user location.
            longitude (float): The longitude of the user location.
        """
        folium.Marker(location=[latitude, longitude],
                    popup=folium.Popup(input_address, 
                                       max_width=250),
                    icon=folium.Icon(icon='fa-location-dot', 
                                     color='red')).add_to(m)


    def add_proximity_circle_to_map(self, m, latitude, longitude, proximity_threshold):
        """
        Add a proximity circle to the map to display boundary.

        Args:
            m (folium.Map): The Folium map.
            latitude (float): The latitude of the user location.
            longitude (float): The longitude of the user location.
            proximity_threshold (float): The proximity threshold in km.
        """
        radius_meters = proximity_threshold * 1000  # Convert proximity threshold from km to meters
        
        folium.Circle(
            location=[latitude, longitude],
            radius=radius_meters,
            color='red',
            fill_color='gray',
            fill_opacity=0.1,
        ).add_to(m)

        # Add the scale control (optional feature)
        measure_control = MeasureControl(position='topright', active_color='blue', primary_length_unit='kilometers')
        m.add_child(measure_control)


    def add_polyline_to_map(self, m, df, latitude, longitude):
        """
        Add a polyline to the map, connecting the input location and the location closest to it.

        Args:
            m (folium.Map): The Folium map.
            df (pandas.DataFrame): The filtered locations DataFrame.
            latitude (float): The latitude of the user location.
            longitude (float): The longitude of the user location.
        """
        if df is not None and not df.empty:
            nearest_locations = df[df['proximity'] == df['proximity'].min()]  # may have > 1 locations with closest proximity to input address
            for _, location in nearest_locations.iterrows():
                nearest_latitude = location['latitude']
                nearest_longitude = location['longitude']

                folium.PolyLine(
                    locations=[[latitude, longitude], [nearest_latitude, nearest_longitude]],
                    color='black',
                    weight=2,
                    opacity=1.0
                ).add_to(m)


    def save_folium_map(self, m, map_filepath):
        """
        Save the Folium map to a file.

        Args:
            address (str): The user input address.
            m (folium.Map): The Folium map.
            map_filepath (str): The filepath to save the map to.
        """
        # Generate a unique filename based on the address
        
        m.save(map_filepath)
        print("Map saved to " + map_filepath)


    def display_map(self, df, input_address, latitude, longitude, proximity_threshold, map_type):
        """
        Display the map with the relevant components.

        Args:
            df (pandas.DataFrame): The filtered locations DataFrame.
            input_address (str): The user input address.
            latitude (float): The latitude of the user location.
            longitude (float): The longitude of the user location.
            proximity_threshold (float): The proximity threshold input by user.
            map_type (str): The type of map to be generated.
        """
        print("Generating map...\n")

        filename = f"map_{input_address.replace(' ', '_')}_{map_type.replace(' ', '_')}.html"
        map_filepath = os.path.join(self.current_dir, "maps", filename)

        m = self.create_folium_map(df, input_address, latitude, longitude, proximity_threshold, map_type)

        print("Map generated!\n")
        
        self.save_folium_map(m, map_filepath)
        self.show_map_in_webview(map_filepath)


    def show_map_in_webview(self, map_filepath):
        """
        Show the map in a web view.

        Args:
            map_filepath (str): The filepath of the map file to display.
        """
        self.web_view.load(QUrl.fromLocalFile(map_filepath))


    def get_zoom_level(self, proximity_threshold):
        """
        Get the zoom level based on the proximity threshold.

        Args:
            proximity_threshold (float): The proximity threshold.

        Returns:
            int: The zoom level.
        """
        # TODO: Change the settings according to the scale of proximity threshold
        min_threshold = 0.1  # Minimum proximity threshold
        max_threshold = 10  # Maximum proximity threshold
        zoom_min_threshold = 17  # Minimum zoom level
        zoom_max_threshold = 12  # Maximum zoom level

        # Calculate the proportional zoom level
        zoom_level = np.interp(proximity_threshold, [min_threshold, max_threshold], [zoom_min_threshold, zoom_max_threshold])

        return int(zoom_level)
    

    def filter_locations(self, location_data, user_latitude, user_longitude, proximity_threshold):
        """
        Filter locations based on proximity to the user location.

        Args:
            location_data (pandas.DataFrame): The location data.
            user_latitude (float): The latitude of the user location.
            user_longitude (float): The longitude of the user location.
            proximity_threshold (float): The proximity threshold input by user.

        Returns:
            pandas.DataFrame: The filtered locations DataFrame.
        """
        filtered_locations = location_data[
            location_data.apply(
                lambda row: distance.distance(
                    (user_latitude, user_longitude),
                    (row['latitude'], row['longitude'])
                ).km <= proximity_threshold,
                axis=1
            )
        ]

        if not filtered_locations.empty:
            # Sort the filtered locations by proximity
            filtered_locations['proximity'] = filtered_locations.apply(
                lambda row: distance.distance(
                    (user_latitude, user_longitude),
                    (row['latitude'], row['longitude'])
                ).km,
                axis=1
            )
            filtered_locations.sort_values(by='proximity', inplace=True)

        return filtered_locations


    def print_addresses(self, df, proximity_threshold):
        """
        Print the filtered addresses.

        Args:
            df (pandas.DataFrame): The filtered locations DataFrame.
            proximity_threshold (float): The proximity threshold.
        """
        addresses = df['address'].tolist()
        count = len(addresses)
        print(f"Number of locations found within {proximity_threshold} km proximity: {count}\n")

        print("-------------------------FILTERED LOCATIONS-------------------------")
        for address in addresses:
            print(address)
        print("--------------------------------------------------------------------\n")


def main():
    app = QApplication([])
    geo_app = GeoApp()
    geo_app.show()
    app.exec_()


if __name__ == "__main__":
    main()