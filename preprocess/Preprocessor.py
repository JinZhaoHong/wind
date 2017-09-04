import sys
import os
import re
import numpy as np

class Preprocessor :
        
    def preprocess(self, folder_name = "Easter_Cape_Cod_MA") :
        '''
        Folder contains many csv files. Each csv file contains position and capacity factors of a wind point.
        preprocess returns a list of position lists of every wind points as well as 
        a list of capacity lists which consists of capacities for every hour
        positions = [[-40, 50], [-45, 39]]
        capacities = [[20, 22, 25, 30, 25, 19], [20, 20, 22, 22, 20, 21]]
        '''
        capacities = []
        positions = []
        match = r".+\.csv"
        for root, subdirs, files in os.walk("./" + folder_name):
            for filename in files:
                # To check if the file is in the format of csv or not
                if (re.match(match, filename) != None) :
                    file_path = os.path.join(root, filename)
                    # Open each file, create a list of features of a wind_point and add it to wind_point_dictionary(dictionary)
                    # Also, create a list named position and append it to positions(list)
                    with open(file_path, 'r') as f:
                        f_contents = f.readlines()
                        capacity = []
                        position = []
                        for counter, line in enumerate(f_contents) :
                            if (counter == 0) :
                                # Line 0 : SiteID - we don't need SiteID
                                pass 
                            elif (counter == 1) :
                                # Line 1 : Longitude - store longitude to a variable (to append lat and lon in order later)
                                lon = float(line.strip().split(",")[-1])
                            elif (counter == 2) :
                                # Line 2 : Latitude - store latitude to a variable and append it to position
                                lat = float(line.strip().split(",")[-1])
                                position.append(lat)
                            elif (counter == 3) :
                                # Line 3 : Column names - Now that we have lat and lon, append lon to position. 
                                # Skip Line 3 since it's just the names of columns 
                                position.append(lon)
                                pass
                            elif ((counter - 4) % 12 == 0) :
                                # append the first "minute" of an hour 
                                capacity.append(float(line.strip().split(",")[-1]))
                            elif ((counter - 4) % 12 == 11) :
                                # add the last "minute" of an hour and divide it by 12 to find the average
                                capacity[-1] += float(line.strip().split(",")[-1])
                                capacity[-1] /= 12
                            else :
                                # add 2nd to 11th "minutes" of an hour
                                capacity[-1] += float(line.strip().split(",")[-1])
                        positions.append(position)
                        capacities.append(capacity)
                        f.close()
        positions = np.array(positions)
        capacities = np.array(capacities)
        return (positions, capacities)


