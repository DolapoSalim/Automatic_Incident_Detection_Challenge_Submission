So we wanna enter a contest and our aim is to win so we have to pull the best resources and best agent for software dev and computer vision, SO READ ALL OF THESE BELOW, SAVE THE INFORMATION, AND FIND THE BEST WAY TO SOLVE IT SO WE CAN WIN!:


The Automatic Incident Detection Challenge 2026 (AID2026) aims to advance the state of the art in real-time accident and anomaly recognition in road-traffic video. Despite significant progress in video understanding, reliable detection of road incidents in unconstrained, real-world conditions remains an open challenge. One of the primary issue hindering further advances is the scarcity of datasets that are explicitly designed for fixed-camera, large-scale traffic monitoring. In fact, most existing datasets are captured using dashcams mounted on vehicles, thus providing an ego-centric viewpoint. Furthermore, datasets that include videos from fixed surveillance cameras are often heterogeneous, as they still combine footage acquired from both dashcams and stationary cameras.

AID26 addresses the above mentioned issues by providing the research community with a carefully designed dataset of 2556 real traffic videos containing annotated incidents captured in real environments (with starting and ending time of each incident), together with a standardized evaluation protocol tailored to real-time deployment constraints.

Within the competition, we invite participants to use the proposed dataset to develop robust and efficient models capable of detecting road incidents using surveillance cameras. Submissions will be evaluated on accuracy, temporal responsiveness, and computational efficiency, reflecting the practical needs of intelligent transportation systems and large-scale video surveillance networks. By benchmarking diverse approaches under a unified framework, AID26 aims to create a reference point for future research on traffic safety and automated monitoring, driving progress toward intelligent systems capable of preventing, detecting, and mitigating incidents in modern smart cities.

The MIVIA-AID dataset is a comprehensive collection of video clips specifically designed to train and evaluate action recognition models for detecting incident. The dataset is completely balanced to ensure unbiased model training and robust evaluation.

2.556
Total Videos
1661
Positive Samples
895
Negative Samples.

Temporal Annotations
Each positive video includes precise timestamp annotations marking the exact moment when incident occurs, enabling fine-grained temporal action localization and early detection training.

Real-World Diversity
Videos captured in diverse real-world conditions: day/night scenarios, multiple lighting conditions, various camera angles, and different environmental contexts to ensure robust model generalization.

Evaluation Protocol
Submitted methods are evaluated according to three key metrics: Precision, Recall, and F1-Score. The test set includes both positive samples (videos containing exactly one incident event) and negative samples (videos with no incident).

Each prediction is evaluated by comparing the ground truth incident start instant 
 with the detection instant 
. The ground truth represents the onset instant where the incident becomes visible, manually annotated after reviewing the complete video. A tolerance of 
 seconds is allowed for early detection. At the same time, we discard detections occuring later than 
 seconds after the ground truth instant.

To compute evaluation metrics, each detection is assigned to one of three categories:

True Positives 
: detections in positive videos that occur within the valid time window, i.e. 
;
False Positives 
: detections occurring in negative videos or in positive videos that happen too early or too late, namely when 
 or 
;
False Negatives 
: the set of positive videos where no detection occurs.


Final Ranking: The contest winner will be determined based on the highest F1-Score achieved on the test set, as it provides the most balanced evaluation of detection accuracy combining both precision and recall performance.

However, to emphasize the applicability of these methods in real-time scenarios, we also compute additional metrics. To find the model with the best performance and the most efficient solution for real-time systems, we consider measures that evaluate both detection speed and computational efficiency. These metrics help identify methods that can operate effectively in resource-constrained surveillance environments while maintaining high detection accuracy.

Notification Delay
Measures how quickly the method detects incident after it occurs. Lower notification delay means faster response, crucial for timely intervention in incident activities. The normalized score transforms the average delay into a value between 0 (worst performance, maximum delay) and 1 (best performance, instantaneous detection).

Processing Frame Rate
The average number of frames processed by the method in one second on a target GPU. Higher PFR indicates faster processing speed, essential for real-time surveillance applications.

Memory Usage
The memory in GB occupied by the method on the target GPU during inference. Lower memory usage enables deployment on resource-constrained edge devices for smart city surveillance.

The methods proposed by the participants will be executed on a private test set. To leave the participants totally free to use all the software libraries they prefer and to correctly reproduce their processing pipeline, the evaluation will be done on Google Colab (follow this tutorial) by running the code submitted by the participants on the samples of our private test set.

Therefore, the participants must submit an archive including the following elements:

A Python script test.py, which takes as input the folder of the test videos (--videos) and produces as output a .CSV file, contained in the folder --results, which can be specified by the user.
The .CSV file must have one column containing incident start instant (in seconds). If no incident is detected in the video the start column must be left empty.
Thus, the script may be executed with the following command:
python test.py --videos foo_videos/ --results foo_results/
A Google Colab Notebook test.ipynb, which includes the commands for installing all the software requirements and executes the script test.py.
All the files necessary for running the test, namely the trained model, additional scripts and so on.
The provided sample test.py includes the writing of the file with the results.
The .CSV file containing the annotations includes a column with the name of the video ('Id Video'), a column with its duration ('Duration') and in case of a positive video (containing an incident), the columns 'Start' and 'End' contains the incident start and end instants, respectively (empty otherwise).

