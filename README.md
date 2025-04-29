SPOTTER (Smart Pose-tracking, Observation, and Training Technique Evaluation in Real-time)

Phillip Williams, Leo Gutierrez

SPOTTER is a full-stack IoT system that enhances the safety and efficacy of physical therapy and workouts by providing real-time feedback on exercise form. The system uses pose estimation and machine learning to monitor users during exercises like pushups and squats, detecting improper form and delivering corrective audio/visual cues. 

Our project architecture is built to capture an image on the client device, push it to AWS for storage, then retrieve and process the image feed on a remote desktop with a GPU. 

While we could have done all the processing on the AWS server, using an external GPU both saves us costs (GPU instances are expensive!) and enables us to process the image feed in real time as the user executes their rep. Additionally, the AWS node is needed to circumvent network security restrictions placed by Duke OIT (which prevented us from pinging the external PC so often).

Once the processing is done, the external PC publishes the pose classification and feedback to the AWS server via a simple JSON API endpoint.

The client device then reads this feedback and activates audiovisual cues to convey this information to the user. 

