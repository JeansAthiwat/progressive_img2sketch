# Good Kid M.a.a.D city is a good album

# Current Pipeline rough idea

We start from an RGB camera image of a single building
 |
\/
Extract the line drawing sketch (Informative Drawings looks like a good model to use)
 |
\/
extract the perspective lines and basic geometric shapes (such as rectangles, triangles, and circles). I believe this should be the first layer of progressive line details, helping the user understand the building’s overall structure in a simplified, abstract form. (not sure if this step is needed if the LOD model is already good)
 |
\/
find some LOD model/algorithm to extract the line drawing sketch
 |
\/
create an iterative interface that can gradually fill in the details from primitive shape and then iterate through the extracted each LOD level