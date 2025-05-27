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
The output consists of two shape files per route, 1. the route itself, 2. the charging stops with history of values.

Currently, only the Mini Cooper SE and Nissan Ariya are supported for route calculations.
However, by specifying the parameters weight, height, width, drag coefficient, auxiliary power consumption, maximum AC/DC charging power, and battery size, routes can be calculated for any electric vehicle.


Some examples:
Route from Nørre Gade 55 in the city
center of Aarhus to Strandby Kirkevej in Esbjerg. The straight-line distance between the
two locations is 130 km. We set the initial state of charge (SoC) to 50%, making charging
along the way unavoidable.
![image](https://github.com/user-attachments/assets/345e3e7f-ee62-4770-805a-f488aa00f639)


