import yaml
import ray.tune as tune

def create_search_space(search_space_config):
    search_space = {}
    for param, options in search_space_config.items():
        if "choices" in options:
            search_space[param] = tune.choice(options["choices"])
        elif "lower" in options and "upper" in options:
            if "integer" in options and options["integer"] == True:
                search_space[param] = tune.randint(options["lower"], options["upper"])
            else:
                search_space[param] = tune.uniform(options["lower"], options["upper"])
        
    return search_space