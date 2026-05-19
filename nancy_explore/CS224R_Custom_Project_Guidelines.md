# CS224R Spring 2026 Deep Reinforcement Learning
## Custom Project Guidelines

## Contents
1. Requirements and Expected Outcome
2. Important Information
    2.1 Timeline and Grading
    2.2 Project Mentors
    2.3 Project Groups
    2.4 Past Projects
    2.5 Contributions
    2.6 Honor Code and AI Tools Policy
    2.7 Submission Format
    2.8 Using Late Days
    2.9 Computing Resources
3. Project Survey (Due 4/22/26)
4. Project Proposal (Worth 4% - Due 5/1/26)
5. Milestone Report (Worth 5% - Due 5/22/26)
6. Poster Presentation (Worth 8% - Due 6/3/26-9:30am-1.45 pm)
7. Final Report (Worth 18% - Due 6/8/26)

## 1 Requirements and Expected Outcome
The final project requires implementing, evaluating, and documenting a novel research study related to one of the main topics of the course.

The novelty should entail at least ONE property below:
* Answering an open-ended question that is unanswered or partially answered in the existing literature. For example, exploring a new method of your own or one left for future works by a prior paper.
* Designing a method that proposes non-trivial modifications at the component or algorithm level with proper justifications. The modifications may not lead to improvements in all dimensions. For example, a method may lead to higher efficiency but degraded performance. However, we expect analysis of the failure modes to be provided.
* Applying a (new or existing) method to an underexplored application domain where the adaptation is non-trivial. For example, adapting an RL algorithm to new environments, where different constraints require certain modification.

For all projects, regardless of the performance, we expect the project to offer new insights that shed light on the success or failure modes of an idea.

The novelty does NOT require:
* Building a new method that meets the standard of machine learning conferences (though some projects may become research papers in the future!).
* Achieving state-of-the-art performance.

In light of the above goals, students are expected to prepare a proposal, prepare a milestone report, present their work in the final poster session, and submit the final report, with specific details as outlined in this document¹.

## 2 Important Information

### 2.1 Timeline and Grading
The project will count for 35% of the course grade in total. Each component of the final project will be due to Gradescope² by 9 p.m on the dates in Figure 1. This document should serve as your go-to resource for any and all information regarding the final project. If and when updates are made to this document, the course staff will notify the entire class immediately and detail the changes.

² With the exceptions of the project survey through Google Forms and the in-person poster session.

**Figure 1: Project Timeline and Grade Distribution.**
* Project survey (Sec. 3): 4/22/26
* Project proposal Grade: 4% (Sec. 4): 5/1/26
* Project milestone Grade: 5% (Sec. 5): 5/22/26
* Poster presentation Grade: 8% (Sec. 6): 6/3/26 9:30am-1.45 pm
* Final report Grade: 18% (Sec. 7): 6/8/26

**Table 1: Poster Session Schedule and Deadlines**

| Date | Time | Event/Deadline | Participants |
| :--- | :--- | :--- | :--- |
| June 3 | 9:00 AM | Gradescope deadline to submit videos | All-CGOE project groups |
| June 3 | 9:00 AM | Gradescope deadline to submit posters | All project groups |
| June 3 | 9.15 AM-9.30 AM | Session A check-in | Session A groups |
| June 3 | 9.30 AM-11.30 AM | Session A presentations | Session A groups |
| June 3 | 11.30 AM-11.45 AM | Session B check-in | Session B groups |
| June 3 | 11.45 AM-1.45 PM | Session B presentations | Session B groups |

### 2.2 Project Mentors
All final projects will be assigned a single mentor CA to serve as a point of contact. Your project mentor should be your primary point of contact throughout the quarter as you work on your project. This individual will also be the person who ultimately grades your project milestone and final reports. Once your project mentor has been assigned, you should not visit other CAs at office hours to discuss your final project since much of your time will likely be spent simply bringing them up to speed; instead, you should visit your project mentor's office hours for project-related issues and challenges.

In your response to the project survey (Section 3), you may choose to indicate specific members of the teaching staff who you think compliment your interests and/or are well-suited to guide your particular proposed project. To help you with this, listed below are the broad specialities of each CA (alphabetical order):
* Abhijnya Bhat: RL, Generative Models
* Alex Nam: RL, LLM
* Anikait Singh: RL, LLM
* Anubha Mahajan: RL, Robot Learning
* Anushree Aggarwal: RL, LLM
* Ethan Hsu: LLM, Computer Use
* Ifdita Hasan Orney: RL, LLM
* Jonathan Yang: Imitation Learning, Robot Learning
* Joonwon Kang: Robot Learning, Imitation Learning, RL, Safety
* Joy He-Yueya: RL, LLM
* Ke Wang: Robotics
* Marcel Torne: RL, Robot Learning
* Max Du: Robot Learning, Imitation Learning, RL
* Megha Srivastava: RL, LLM, HRI
* Perry Dong: RL
* Rahul Chand: RL, LLM, Robotics
* Riya Karumanchi: RL
* Shengqu Cai: Generative Models, Infra
* Tian Gao: Robot Learning, Imitation Learning, RL
* Tyler Lum: RL, Robot Learning

Clearly, this list of reinforcement-learning topics is not exhaustive and, furthermore, each CA likely has expertise that extends beyond what is listed. That said, even if you find yourself matched with a project mentor whose specialities do not overlap with your chosen project area, your mentor is still broadly well-versed in reinforcement learning, well aware of what constitutes a solid final project for the course, and is ready to help you to the best of their abilities. We anticipate that some final projects will fall outside our areas of expertise as a teaching staff, and that is okay so long as there is clear connection to and application of core course topics.

You should think of your project mentor as an individual who can provide you with feedback as you hit checkpoints in your final project; it would not be in your best interest to seek out advising from your project mentor on a week-by-week basis. Instead, consult your mentor when you have uncertainty on the path ahead or there are major decisions to be made which can dramatically impact the contents of your final report.

### 2.3 Project Groups
Final projects for the course may be completed by individual students or in teams of no more than 3 people. We encourage everyone to work in groups with the expectation that the overall contributions of the final project be commensurate with the group size. While the project survey will ask that your group for CA matching purposes, project groups are mutable up to and until the deadline of the project proposal. In other words, your group is locked in once you submit your project proposal.

### 2.4 Past Projects
For inspiration, you can browse examples of final projects from previous offerings of the course: Spring 2025 Final Projects.

### 2.5 Contributions
To help us (and yourselves) keep track of individual workloads, we ask that project proposals include an anticipated breakdown of individual roles and responsibilities. We will also ask for your final report to include a similar breakdown of individual contributions as well. It is perfectly fine for roles to adapt and change over the course of the project including deleted roles and newly added roles between the proposal and final report. If there are any changes in responsibilities, we ask that they be documented in the final report so we can ensure an equitable division of labor among group members. Project groups are welcome to work with people not enrolled in the class, however we expect the contribution breakdowns in the project proposal and final report to reflect the work of all members involved (both internal and external). Final projects may be shared between CS224R and another class however, in this case, you should clearly indicate this in your project proposal and we will expect a more ambitious project.

### 2.6 Honor Code and AI Tools Policy
You should work as a team independently and ensure academic integrity by clearly documenting which parts of your code or ideas were assisted by AI tools and which parts were developed independently.

For the custom project: You are allowed to collaborate with AI tools such as GitHub Co-Pilot and ChatGPT. These tools can help you generate boilerplate code and set up tools like a dataloader.

Disclosure Requirement: Both the milestone report and final report must include a brief AI Tools Disclosure section describing how Al tools were used during the project. This should cover which tools were used (eg, ChatGPT, GitHub Copilot, Claude), what they were used for (eg, boilerplate code, debugging, writing assistance), and which parts of the project were developed independently. This section will not affect your grade but is required for academic integrity. However, to maximize learning through active implementation, we expect that you use them only as an aid rather than the complete solution. For example, it is acceptable to leverage AI for standard coding tasks (e.g., constructing a data pipeline or debugging minor code issues). For essential components of your project, such as implementing deep reinforcement learning algorithms (e.g., Proximal Policy Optimization (PPO)), you should write the code independently. This ensures that you develop a deep understanding of the underlying principles and techniques. Every custom project team has a mentor, who gives feedback and advice during the project. Note: TAs may not look at students' code for either the default or custom final projects.

### 2.7 Submission Format
We recommend submitting proposals and reports using the provided latex templates below. This ensures consistent formatting for all submissions.
* Project Proposal: CS224R Project Proposal Template
* Milestone: CS224R Project Milestone Template
* Final Project: CS224R Final Report Template

For poster presentations, you are free to choose templates that effectively communicate your research findings, such as Stanford-LaTeX-Poster-Template. All submissions must adhere to the specified page limits in the instructions.

### 2.8 Using Late Days
With the exception of the project poster (which cannot be submitted late due to the live poster session) and the final project report (which cannot be submitted late due to the University deadline to submit final grades for the quarter), entire project groups may apply a maximum of two late days to any other graded component of the final project, for example, project milestone. Groups cannot aggregate collective late days and late days will be applied to all group members; this means that, to apply two late days, each group member must have two unused late days to spare. Assignment extensions granted by the Office of OAE do not apply to group assignments. However, if you are working individually they can apply.

### 2.9 Computing Resources
We will be providing $500 in cloud computing credits to each student for use on homework assignments and the final project, which should amount to more than 200 hours of GPU time (and much more CPU time), depending on the size of the GPU. Details about how to access these credits are forthcoming.

## 3 Project Survey (Due 4/22/26)
All projects should evaluate novel ideas that pertain to deep reinforcement learning, potentially spanning foundational algorithm design to novel applications. For students conducting research in a lab on campus, you are encouraged to pursue a project related to your research area, but are not required to. You should think early about the data (simulated or real) that you'll need to collect, and the computational resources you'll need.

**Brainstorming** - You may discuss the topic of your final project with course staff by private message on Ed or in office hours. If you are not sure about the topic, we encourage you to speak with us. If you are looking for project ideas, we will soon post a document on Ed that contains some ideas we have collected from the AI community, though we also encourage you to come up with your own.

**Evaluating Ideas** - Here are some examples of weak project proposals that do not satisfy the project requirements, and how they can be improved:
1. Weak: run an existing algorithm out of the box on a new dataset. Strong: develop a modified algorithm that is particularly suited for the challenges of a new application
2. Weak: re-implement an existing reinforcement-learning algorithm. Strong: re-implement a recent paper and investigate an extension of the method that may have been mentioned as future work in the paper.
3. Weak: sweep hyper-parameters, do architecture search of some algorithm. Strong: investigate the weaknesses of a particular algorithm when tested in new ways and pursue a solution

**Submission** - Everyone, either individually or as part of a larger group, should submit the Project Survey which will allow us to collect preliminary information on final project topics, determine the total number of groups, and assign project mentors. While project groups should, ideally, be finalized by this stage, we will allow group changes up to the project proposal deadline.

**Mentor Assignments** - The course staff will release a spreadsheet detailing project mentor assignments for each group within a few days of the survey submission deadline.

## 4 Project Proposal (Worth 4% – Due 5/1/26)
The project proposal should be an extended abstract motivating and outlining the project you plan to complete. Your proposal should have the following structure:
1. Objective 1/4 page. Explain the objectives of the project and why the objectives are relevant and important.
2. Related Work 1/2 page. Review the most relevant prior work, and highlight where those works fall short of meeting the objectives described above.
3. Technical Outline 1/2 page. Explain your approach at a high-level, making clear the novel technical contribution, and describe the evaluation plan.
4. Team Contributions 1/4 page. Include a brief, high-level breakdown of the planned contributions of each individual group member.

**Submission & Grading** - Submit one proposal PDF per group to Gradescope under the "Project Proposal" assignment that includes the (finalized) list of project group members. While the deadline for the project proposal is in late-April, we strongly encourage you to begin developing your idea for the project sooner. The grading for the proposal will be primarily intended to give you feedback on how to adjust/pivot your proposal to meet the project requirements.

**Modifying Contributions** - It is perfectly fine for items in your breakdown of team member contributions to remain, adapt, or even disappear between submitting the proposal and the final project. Our goal is to get as even a distribution of workload as possible amongst group members. Any changes to this list of contributions that you do make after submitting the proposal should be documented in your final project report.

## 5 Milestone Report (Worth 5% – Due 5/22/26)
Your milestone report should be one-page document that answers the following questions:
1. What experiments have you conducted so far? Tell us about these experiments.
2. Are there any changes to the research hypothesis or objective from the proposal based on your initial findings?
3. What are the concrete steps that need to happen in order to bring the project from where it is now to completion?

**Required Experiment** - The milestone report must provide details on at least one experiment that you have done since the proposal. The experiment need not be successful, but you should have attempted something. If it did not work as expected, you should briefly discuss why. You are encouraged to include figures.

**Submission & Grading** - Submit one one-page milestone report PDF per group, with the names of all project members, to Gradescope under the assignment "Milestone Report." The milestone report will be graded by your project mentor. The grading of the milestone report will be light as the primary goal is to provide you feedback so that there is a clear path going forward to the poster presentation and final report.

## 6 Poster Presentation (Worth 8% - Due 6/3/26-9:30am-1.45 pm)
Students are expected to prepare a research poster that briefly but structurally answers the following questions:
1. What is the problem?
2. Why is the problem interesting and important?
3. Why hasn't it been solved before? Or, what's wrong with previous proposed solutions? (i.e. introduce relevant prior works)
4. What are the key components of your approach?
5. What are the key findings from my experiments?
6. What can audiences learn from this work? What are the limitations or directions for future work?
7. * The poster should also include any work that is planned to be done before completion of the final report.

**Poster Dimensions** - We recommend printing posters in landscape orientation with size 24" x 36". While other sizes are acceptable, this size will work best with the poster boards and easels we will provide for you at the poster session, so please make sure your poster isn't too much bigger or smaller than this.

**Poster Printing** - Please note that you are not required to do commercial poster printing; you can print the poster on smaller sheets (letter size or A4 size) using regular printers and tape them together. Adobe Acrobat Reader automates this - look for the Poster options in the Print dialog. That said, both FedEx and Walgreens have deals where, if it's your first time printing or you make a new account with another email, you can print a poster for around $15-20. You are expected to print out the posters yourself, so please plan ahead! Some other options that have been suggested in other CS courses include: Lathrop Library's Tech Desk, CVS Photo, Biotech Productions, and Walmart.

**Poster Exemptions** - At least one group member must be present at the poster session, unless you have reached out via email to the course staff and received special permission to submit a video presentation. The only exception is if the project group is made up entirely of (remote) CGOE students, in which case, the group will be expected (by default) to submit a five-minute video presentation of their poster instead. A Google Drive or YouTube link pointing to the video should be uploaded to Gradescope under the "Project Poster Video" assignment.

**Location & Time** - The poster session will be held at the ACSR Basketball courts with the following two sessions: Session A: 9:30am-11:30am (check in 9:15am-9:30am); Session B: 11:45am-1:45pm (check in 11:30am-11:45am).

## 7 Final Report (Worth 18% – Due 6/8/26)
The final report should be in the style of a research paper, containing two parts:
1. A one-page extended abstract: The one-page extended abstract should summarize the main findings and accomplishments of your final project. The extended abstract should be attached as the first page of the full report.
2. The main paper: It should describe and motivate the method in detail, discuss relevant prior work and how the project makes new contributions relative to these prior works, and present and discuss the results, including any relevant figures or plots.

**Formatting & Contributions** - Successful reports will have a main body that is about eight pages in length, but there are no hard length requirements or formatting requirements, except the one-page extended abstract. For project groups, please include a fully-updated breakdown of the contributions of individual team members and highlight why these adjustments were necessary from your original breakdown in the project proposal; this is as much a reflection on the research process for you as it is a tool for us to track equitable division of labor.

**Submission & Grading** - Submit one final report per group with the names of all project members to Gradescope under the assignment "Final Project Report."
