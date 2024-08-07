# 

Often times when debugging on a single node, we have direct access to the process, we can see all the output, we can see the stacktrace if the program is hanging and we kill it, etc.

We have all the information! This makes debugging a problem a breeze.

When in a distributed setting this isn't the case. We have to carefully set up all of the logging so we have all of that information, in case a crash or hang happens.
