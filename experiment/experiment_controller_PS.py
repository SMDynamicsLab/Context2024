# -*- coding: utf-8 -*-
"""
Created on Jun 2022

@author: RLaje, paulac, ASilva
"""

import serial, time
import numpy as np
import numpy.matlib as mlib
import random
import os
import pandas as pd
import json
import tappingduino as tp


#%% Description

#==============================================================================
# Saves:  - a file per trial containing the raw data from it.
#         - a file per trial containing extracted data from it.
#         - a file per block containing information about all trials in it.
#==============================================================================

#%% Define Python user-defined exceptions.
class Error(Exception):
	"""Base class for other exceptions"""
	pass


#%% Definitions.

# Define variables.

ISI = 500                     	                                                                      # Interstimulus interval (milliseconds).
n_stim = 35                                                                                           # Number of bips within a sequence.
n_trials_percond = 12		                                                                          # Number of trials per condition.
n_blocks = 3                 	                                                                      # Number of blocks.
n_subj_max = 100             	   	                                                                  # Maximum number of subjects.
perturb_bip_range = (17,22)		                                                                      # Perturbation bip range.

perturb_type_dictionary = {'PS':1}      														      # Perturbation type dictionary. 0--> Step change. 1--> Phase shift.                                     
perturb_size_dictionary = {'neg':-50, 'neg2':-20, 'iso':0, 'pos2':20, 'pos':50}					      # Perturbation size dictionary.

condition_dictionary_df = tp.Condition_Dictionary(perturb_type_dictionary, perturb_size_dictionary)   # Possible conditions dictionary.
n_conditions = len(condition_dictionary_df) 														  # Number of conditions.

perturb_type_perCondition_list = condition_dictionary_df['Perturb_type'].tolist()                     # Relationship between conditions and perturb type.
perturb_size_perCondition_list = condition_dictionary_df['Perturb_size'].tolist() 					  # Relationship between conditions and perturb size.

# Number of trials.
n_trials = n_trials_percond * n_conditions
if (n_trials % n_blocks == 0):
	n_trials_perblock = n_trials // n_blocks 
else:
	print('Error: Número de trials no es múltiplo del número de bloques.')
	raise Error

#path = '../data/Experiment_PS_SC/'
path = '../data/Experiment_PS/'
#path = '../data/Experiment_SC/'

# Save experiment parameters.
experiment_parameters_dictionary = {'ISI' : ISI, 'n_stim' : n_stim, 'n_trials_percond' : n_trials_percond, 'n_blocks' : n_blocks, 
                                    'n_subj_max' : n_subj_max, 'perturb_bip_range' : perturb_bip_range, 'perturb_type_dictionary' : perturb_type_dictionary,
                                    'perturb_size_dictionary' : perturb_size_dictionary, 'n_conditions' : n_conditions, 'n_trials' : n_trials,
                                    'path' : path}

with open(path + 'Experiment_parameters.dat', 'w') as fp:
    json.dump(experiment_parameters_dictionary, fp)


#%% Conditions

# Filename for the file that will contain all possible permutations for the subjects.
presentation_orders = path + 'Presentation_orders.csv'

# Start experiment or generate Perturbation_orders.csv file.
start_or_generate_response = input("Presione enter para iniciar experimento, o escriba la letra G (generar archivo con órdenes de presentación) y presione enter: ") 

# If this is the first time running the experiment, then it's necessary to generate the Presentation_orders.csv file.
if start_or_generate_response == 'G':
	confirm_response = input('¿Está seguro? Si el archivo ya existe se sobrescribirá. Escriba S y presione enter para aceptar, o sólo presione enter para cancelar: ')
	
	if confirm_response == 'S':
		chosen_conditions = mlib.repmat(np.arange(0,n_conditions),n_subj_max,(n_trials_percond))
		for i in range(0,n_subj_max):
			random.shuffle(chosen_conditions[i])

		presentation_orders_df = pd.DataFrame()
		
		for i in range(0,n_subj_max):
			next_subject_number = '{0:0>3}'.format(i)
			next_subject_number = 'S' + next_subject_number
			presentation_orders_df[next_subject_number] = chosen_conditions[i]
        
		presentation_orders_df.index.name="Trial"

		presentation_orders_df.to_csv(presentation_orders)


else:

    # Communicate with arduino.
    #arduino = serial.Serial('COM4', 9600)
    arduino = serial.Serial('/dev/ttyACM0', 9600)
	   
    # Open Presentation_orders.csv file as dataframe.
    presentation_orders_df = pd.read_csv(presentation_orders,index_col='Trial')
	
	
    # EXPERIMENT

    # Check for file with names and pseudonyms.
    filename_names = path + 'Dic_names.dat'
    filename_names_pseud = path + 'Dic_names_pseud.dat'
	
    cont_exp = input('\nSi necesita retomar un experimento anterior, escriba S y presione enter. De lo contrario, presione enter para continuar: ')
    if (cont_exp == 'S'):
        current_subject_number = input('\nIngrese el número de sujeto. De lo contrario presione enter para seleccionar el último sujeto por defecto: ')
        if (current_subject_number  == ""):
            try:
                f_names = open(filename_names,"r")
			
                content = f_names.read()
                last_subject_number = int(content [-3:])
                curr_subject_number_int = last_subject_number
                curr_subject_number = '{0:0>3}'.format(curr_subject_number_int)
                f_names.close()
            except IOError:
                print('El archivo no esta donde debería, o no existe un sujeto previo.')
                raise
        else:
            try:
                f_names = open(filename_names,"r")
                
                content = f_names.readlines()
                data_lines = []
                curr_subject_number = ""
                for data_line in content:
                    data_lines.append(data_line.strip('\n'))
                for data_line in data_lines:
                    if (data_line != ""):
                        last_subject_number = int(data_line [-3:])
                        curr_subject_number_int = last_subject_number
                        if (curr_subject_number_int == int(current_subject_number)):
                            curr_subject_number = '{0:0>3}'.format(curr_subject_number_int)
                if (curr_subject_number == ""):
                    print('No se encuentra al sujeto.')
                    raise Error
                curr_subject_number_int = int(current_subject_number)
            except IOError:
                print('El archivo no esta donde debería, o no existe un sujeto previo.')
                raise
					
        current_block_counter = input('\nIngrese el número de bloque deseado y presione enter. De lo contrario, presione enter para seleccionar 0 por defecto: ')
        if (current_block_counter != ""):
            if (int(current_block_counter) >= 0 and int(current_block_counter) < n_blocks):
				# Run blocks.
                block_counter = int(current_block_counter)
                tp.Unfinished_Files(path, curr_subject_number, block_counter, n_blocks)
            else:
                print('El número de bloque seleccionado no es válido.')
                raise Error
        else:
			# Run blocks.
            block_counter = 0
            tp.Unfinished_Files(path, curr_subject_number, block_counter, n_blocks)


    else:
        try:
            f_names = open(filename_names,"r")

            if os.stat(filename_names).st_size == 0:
                curr_subject_number_int = 0
                curr_subject_number = '{0:0>3}'.format(curr_subject_number_int)
                f_names.close()
            else:
                content = f_names.read()
                last_subject_number = int(content [-3:])
                curr_subject_number_int = last_subject_number + 1
                curr_subject_number = '{0:0>3}'.format(curr_subject_number_int)
                f_names.close()
            
        except IOError:
            print('El archivo no está donde debería, ubicarlo en la carpeta correcta y volver a correr esta celda.')
            raise
			
		# Set subject name for filename.
        name = input("Ingrese su nombre: ") 
        f_names = open(filename_names_pseud,"a")
        f_names.write('\n'+name+'\tS'+curr_subject_number)
        f_names.close()
        f_names = open(filename_names,"a")
        f_names.write('\nS'+curr_subject_number)
        f_names.close()

	    # Run blocks.
        block_counter = 0


	# Trials for the current subject.
    subject_df = pd.DataFrame(presentation_orders_df['S' + curr_subject_number])
    subject_df.rename(columns={'S' + curr_subject_number:'Condition'},inplace=True)


    while (block_counter < n_blocks):

		# Block conditions.
        block_conditions_aux = block_counter * n_trials_perblock
        block_conditions_df = (subject_df.loc[block_conditions_aux : block_conditions_aux + n_trials_perblock - 1]).reset_index()
        block_conditions_df = block_conditions_df.drop(columns = ['Trial'])
        block_conditions_df.index.name="Trial"
        block_counter_list = []
        perturb_bip_list = []
        perturb_size_list = []
        perturb_type_list = []
        for i in range(0,n_trials_perblock):
            block_counter_list.append(block_counter)
            perturb_bip_list.append(random.randrange(perturb_bip_range[0],perturb_bip_range[1],1))
            condition_type = (block_conditions_df.loc[[i]].values.tolist())[0][0]
            perturb_type_list.append(perturb_type_perCondition_list[condition_type])
            perturb_size_list.append(perturb_size_perCondition_list[condition_type])
        block_conditions_df = block_conditions_df.assign(Block = block_counter_list, Original_trial = range(0,n_trials_perblock), 
            Perturb_bip = perturb_bip_list, Perturb_size = perturb_size_list, Perturb_type = perturb_type_list)
        block_conditions_df = block_conditions_df.reindex(columns=['Block','Original_trial','Condition','Perturb_type','Perturb_size','Perturb_bip'])


		# Run one block.
        input("Presione Enter para comenzar el bloque (%d/%d): " % (block_counter,n_blocks-1))
		
		# Set time for file name.
        timestr = time.strftime("%Y_%m_%d-%H.%M.%S")

        trial = 0
		
        messages = [] # Vector that will contain exact message sent to arduino to register the conditions played in each trial.
        valid_trials = [] # Vector that will contain 1 if the trial was valid or 0 if it wasn't.
        errors = [] # Vector that will contain the type of error that ocurred if any did.    
        
        # Generate filename for file that will contain all conditions used in the trial along with the valid_trials vector.
        filename_block = path + 'S'+curr_subject_number+"-"+timestr+"-"+"block"+str(block_counter)+"-trials.csv"
        
        while (trial < len(block_conditions_df.index)):
            input("Presione Enter para comenzar el trial (%d/%d):" % (trial,len(block_conditions_df.index)-1))

			# Generate raw data file.
            filename_raw = path + 'S'+curr_subject_number+"-"+timestr+"-"+"block"+str(block_counter)+"-"+"trial"+str(trial)+"-raw.dat"
            f_raw = open(filename_raw,"w+")
         
            # Generate extracted data file name (will save raw data, stimulus time, feedback time and asynchrony).
            filename_data = path + 'S'+curr_subject_number+"-"+timestr+"-"+"block"+str(block_counter)+"-"+"trial"+str(trial)+".dat"    
            
            # Wait random number of seconds before actually starting the trial.
            wait = random.randrange(10,20,1)/10.0
            time.sleep(wait)
            
            # Define stimulus and feedback condition for this trial.
            perturb_size_aux = (block_conditions_df.loc[[trial],['Perturb_size']].values.tolist())[0][0]
            perturb_bip_aux = (block_conditions_df.loc[[trial],['Perturb_bip']].values.tolist())[0][0]
            perturb_type_aux = (block_conditions_df.loc[[trial],['Perturb_type']].values.tolist())[0][0]

			# Send message with conditions to arduino.
            message = str.encode(";S%c;F%c;N%c;A%d;I%d;n%d;P%d;B%d;T%d;X" % ('B', 'B', 'B', 125, ISI, n_stim, perturb_size_aux, perturb_bip_aux, perturb_type_aux))
            arduino.write(message)
            messages.append(message.decode())

			# Read information from arduino.
            data = []
            aux = arduino.readline().decode()
            while (aux[0]!='E'):
                data.append(aux)
                f_raw.write(aux) # save raw data
                aux = arduino.readline().decode()

			# Separates data in type, number and time.
            e_total = len(data)
            e_type = []
            e_number = []
            e_time = []
            for event in data:
                e_type.append(event.split()[0])
                e_number.append(int(event.split()[1]))
                e_time.append(int(event.split()[2]))

			# Separates number and time according to if it comes from stimulus or response.
            stim_number = []
            resp_number = []
            stim_time = []
            resp_time = []
            for events in range(e_total):
                if e_type[events]=='S':
                    stim_number.append(e_number[events])
                    stim_time.append(e_time[events])

                if e_type[events]=='R':
                    resp_number.append(e_number[events])
                    resp_time.append(e_time[events])

			# Close raw data file.    
            f_raw.close()

			# ---------------------------------------------------------------
			# Asynchronies calculation.

			# Vector that will contain asynchronies if they are calculated.
            asyn_df = tp.Compute_Asyn(stim_time,resp_time)
            error_label, error_type, valid_trial_eh = tp.Error_Handling(asyn_df, resp_time)
            errors.append(error_type)
            valid_trials.append(valid_trial_eh)
            if (valid_trial_eh == 0):
                print(error_label)
				# Add 1 to number of trials per block since will have to repeat one.
                block_conditions_df = pd.concat([block_conditions_df, block_conditions_df.iloc[trial].to_frame().T]).reset_index(drop = True)
                block_conditions_df.index.name="Trial"
 
			# SAVE DATA FROM TRIAL (VALID OR NOT).
            f_data_dict = {'Data' : data, 'Stim_time' : stim_time, 'Resp_time' : resp_time, 'Asynchrony' : asyn_df['asyn'].tolist(), 'Stim_assigned_to_asyn' : asyn_df['assigned_stim'].tolist()}   
            f_data_str = json.dumps(f_data_dict)
            f_data = open(filename_data, "w")
            f_data.write(f_data_str)
            f_data.close()

	#==============================================================================
	#         # If you want to show plots for each trial.
	#         plt.show(block=False)
	#         plt.show()
	#         plt.pause(0.5)
	#==============================================================================

			# Go to next trial.
            trial = trial + 1

		
        print("Fin del bloque!\n")


		# SAVE DATA FROM BLOCK (VALID AND INVALID TRIALS, MESSAGES AND ERRORS).    
        block_conditions_df = block_conditions_df.assign(Valid_trial = valid_trials, Message = messages, Error = errors)
        block_conditions_df.insert(1, 'Subject', curr_subject_number_int)
        block_conditions_df.to_csv(filename_block)

		# Go to next block.
        block_counter = block_counter + 1
    
    print("Fin del experimento!")
    arduino.close()

