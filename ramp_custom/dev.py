try:
    from line_profiler import LineProfiler

    def do_profile(follow=[]):
        """decorator for line profiling.

        Parameters
        ----------
        follow : list
            list of methods (actual methods, not their name) to measure
            alongside the main profiled one

        """

        def inner(func):
            def profiled_func(*args, **kwargs):
                try:
                    profiler = LineProfiler()
                    profiler.add_function(func)
                    for f in follow:
                        profiler.add_function(f)
                    profiler.enable_by_count()
                    return func(*args, **kwargs)
                finally:
                    profiler.print_stats(output_unit=1)

            return profiled_func

        return inner


except ImportError:

    def do_profile(follow=[]):
        "Helpful if you accidentally leave in production!"

        def inner(func):
            def nothing(*args, **kwargs):
                return func(*args, **kwargs)

            return nothing

        return inner
