# kernels
# if (no. of support vectors / no. of samples) > approx(0.2)
# You might have done some overfitment and are probably using a wrong kernel.
# This does not mean that the data isn't linearly separable (it might not be)
# Eg: 52% accuracy and 80% data are support vectors, there is defenitely something wrong (overfitment)
# Maybe try a different kernel
# Eg: 52% accuracy and 8% are support vectors, your data might just probably not work out. You can try diferent kernels, but probably, 
# something else is wrong
# A soft-margin support vector can be used in place maybe (soft margin classifiers are defaults for most libraries)
# It allows a degree of error for support vectors
# We introduce a leeway called 'slack'
# A slack of 0 becomes a hard margin
# We want to minimize the slack
# If we raise the value of c, we are saying that we want less violations
# If we lower c, we will be more allowing, for violations
# The purpose of a soft-margin classifier is to prevent overfitting of data

# To classify more than one classes using SVM
# 1. OVR: One versus rest (Used more often)
# 2. OVO: One versus one (Involves more processing but is more balanced)