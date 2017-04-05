

###############################################################################
#                                                                             #
#                   Theano Shared Variables Registry                          #
#                                                                             #
###############################################################################


class SharedsRegistry(object):

    def __init__(self):
        self.vars = list()
        self.names = list()
        self.avg_fs = list()
        self.inv_n_parallel = 1

    def register_func(self, f):
        for var in f.get_shared():
            self.register(var)

    def register(self, var):
        import theano
        import theano.tensor as T
        if var not in self.vars:
            self.vars.append(var)
            self.names.append(var.name)  # (could be None)
            if "int" in var.dtype:
                self.avg_fs.append(None)  # (labmda x: x, ?)
            else:
                y = T.scalar('avg_fac', dtype=var.dtype)
                avg_f = theano.function([y], updates=[(var, var * y)])
                self.avg_fs.append(avg_f)

    def get_ID(self, var_or_name):
        if var_or_name is None:
            raise TypeError("Cannot find using NoneType.")
        try:
            return self.vars.index(var_or_name)
        except ValueError:
            pass
        try:
            return self.names.index(var_or_name)
        except ValueError as exc:
            raise exc("Unrecognized shared var or name: ", var_or_name)

    def get_IDs(self, vars_or_names):
        if not isinstance(vars_or_names, (list, tuple, dict)):
            vars_or_names = (vars_or_names,)
        var_IDs = list()
        for var in vars_or_names:
            var_IDs.append(self.get_ID(var))
        if len(set(var_IDs)) != len(var_IDs):
            raise ValueError("Redundant variables provided.")
        return tuple(var_IDs)

    def get_var(self, var_or_name):
        if var_or_name is None:
            raise TypeError("Cannot find using NoneType.")
        if var_or_name in self.vars:
            return var_or_name
        else:
            try:
                return self.vars[self.names.index(var_or_name)]
            except ValueError as exc:
                raise exc("Unrecognized shared var or name: ", var_or_name)

    def get_vars(self, vars_or_names):
        varbles = list()
        for var in vars_or_names:
            varbles.append(self.get_var(var))
        if len(set(varbles)) != len(varbles):
            raise ValueError("Redundant variables provided.")
        return tuple(varbles)

    def get_vars_from_IDs(self, IDs):
        return [self.vars[i] for i in IDs]

    def get_array(self, idx):
        """ Re-reference the variable in case GPU allocation has changed. """
        return self.vars[idx].container.data

    def set_n_parallel(self, n_parallel):
        self.inv_n_parallel = 1 / n_parallel

    def call_avg_fs(self, var_IDs, avg_fac=None):
        avg_fac = self.inv_n_parallel if avg_fac is None else avg_fac
        for var_ID in var_IDs:
            self.avg_fs[var_ID](avg_fac)

