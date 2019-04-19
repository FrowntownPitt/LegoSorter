package org.firstinspires.ftc.teamcode;

import com.qualcomm.robotcore.eventloop.opmode.Autonomous;
import com.qualcomm.robotcore.eventloop.opmode.LinearOpMode;
import com.qualcomm.robotcore.hardware.DcMotor;
import com.qualcomm.robotcore.hardware.DcMotorSimple;

@Autonomous(name="Belt Test", group="Tests")
public class BeltTest extends LinearOpMode {

    private DcMotor belt;
    private double forwardPower = .3;
    private double reversePower = -.3;

    @Override
    public void runOpMode() {
        belt = hardwareMap.get(DcMotor.class, "beltDrive");
        belt.setMode(DcMotor.RunMode.STOP_AND_RESET_ENCODER);
        belt.setMode(DcMotor.RunMode.RUN_WITHOUT_ENCODER);

        belt.setDirection(DcMotorSimple.Direction.REVERSE);

        waitForStart();
        belt.setPower(forwardPower);

        while(opModeIsActive()){
            if(belt.getCurrentPosition() > 1000){
                belt.setPower(reversePower);
            } else if(belt.getCurrentPosition() < 0){
                belt.setPower(forwardPower);
            }

            telemetry.addData("Belt Position", belt.getCurrentPosition());
            telemetry.update();
        }
    }
}
