import SoloPy as solo
import time

from threading import Thread 

currentLimit=6

pwmFrequency=30
numberOfPoles=24
Vref=500
speedControllerKp=0.15
speedControllerKi=0.005
thresholdold=-1
currentCounter=0

    
def motor_startup_torque():
    global mySolo
    mySolo = solo.SoloMotorControllerUart("COM13")



    #Initial Configuration
    mySolo.set_output_pwm_frequency_khz(pwmFrequency)
    mySolo.set_motor_poles_counts(numberOfPoles)
    mySolo.set_feedback_control_mode(solo.FEEDBACK_CONTROL_MODE.SENSOR_LESS)
    mySolo.set_motor_type(solo.MOTOR_TYPE.BLDC_PMSM)
    mySolo.set_speed_controller_kp(speedControllerKp)
    mySolo.set_speed_controller_ki(speedControllerKi)
    mySolo.set_control_mode(solo.CONTROL_MODE.SPEED_MODE)
    mySolo.set_command_mode(solo.COMMAND_MODE.DIGITAL)

    mySolo.motor_parameters_identification(solo.ACTION.START)
    #print("Identifying the Motor")
    time.sleep(2)
    
    
    
def motor_controller(ThresholdBiceps):
    global Vref, thresholdold
    global currentLimit,currentCounter

    actualMotorcurrent, error = mySolo.get_quadrature_current_iq_feedback()
    print(actualMotorcurrent)
    if abs(actualMotorcurrent) >= currentLimit:
        currentCounter = currentCounter + 1
        if currentCounter >= 3:
            Vref=0
            mySolo.set_speed_reference(Vref)
    else:
        currentCounter=0

    if ThresholdBiceps == 1 and ThresholdBiceps != thresholdold:
        mySolo.set_motor_direction(solo.DIRECTION.CLOCKWISE)							
        print("Biceps activ")
        Vref=500
        mySolo.set_speed_reference(Vref)									#set speed reference in [rpm]
    if ThresholdBiceps == 0 and ThresholdBiceps != thresholdold:
        mySolo.set_motor_direction(solo.DIRECTION.COUNTERCLOCKWISE)
        print("Biceps not activ")
        Vref=500
        mySolo.set_speed_reference(Vref)
    if (ThresholdBiceps == 1 or ThresholdBiceps == 0) and ThresholdBiceps == thresholdold:
        print("Function executed, no new value") 
    #else:
    #    print("No change")
    thresholdold=ThresholdBiceps
    
    

