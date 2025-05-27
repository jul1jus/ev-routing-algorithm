# ev-routing-algorithm
Feasible and efficient routes, even with only one strictly Pareto-optimal path  
High-quality routing for both short and long distances (incl. multiple charging stops)  
More diverse and realistic results than many commercial route planners  

Test data is a combination of OSM road network data + charging station data merged as well as topography data.
The Graph/Slovenia Dir is test Data that does not exceed the 100MB constraint of Github.  
The Data Preprocessing Pipeline:
![image](https://github.com/user-attachments/assets/afde527a-f8d2-4f1d-b67f-cddf6d503a5d)

The Pareto objectives are:  
• Distance: Total distance traveled  
• Time: Total time traveled  
• Energy consumed: Total energy consumed  
• Charging Time: Total time spent charging

The program can be called using python routing_simple
