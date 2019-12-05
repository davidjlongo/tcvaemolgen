import logging

from collections import defaultdict

log = logging.getLogger(__name__)

class ProviderList(object):
    AZURE = 'azure'
    LOCAL = 'local'
    
class ProviderFactory(object):
    def __init__(self):
        self.provider_list = defaultdict(dict)
        log.debug("Providers List: %s", self.provider_list)
        
    def register_provider_class(self, cls):
        if isinstance(cls, type) and issubclass(cls, CloudProvider):
            if hasattr(cls, "PROVIDER_ID"):
                provider_id = getattr(cls, "PROVIDER_ID")
                if self.provider_list.get(provider_id, {}).get('class'):
                    log.warning("Provider with id: %s is already "
                                "registered. Overriding with class: %s",
                                provider_id, cls)
                self.provider_list[provider_id]['class'] = cls
            else:
                log.warning("Provider class: %s implements CloudProvider but"
                            " does not define PROVIDER_ID. Ignoring...", cls)
        else:
            log.debug("Class: %s does not implement the CloudProvider"
                      "  interface. Ignoring...", cls)
            
    def discover_providers(self):
        for _, modname, _ in pkgutil.iter_modules(providers.__path__):
            log.debug("Importing provider: %s", modname)
            try:
                self._import_provider(modname)
            except Exception as e:
                log.warn("Could not import provider: %s", e)
        
    def _import_provider(self, module_name):
        log.debug("Importing providers from %s", module_name)
        module = importlib.import_module(
            "{0}.{1}".format(providers.__name__,
                             module_name))
        classes = inspect.getmembers(module, inspect.isclass)
        for _, cls in classes:
            log.debug("Registering the provider: %s", cls)
            self.register_provider_class(cls)    
        
    def list_providers(self):
        if not self.provider_list:
            self.discover_providers()
        log.debug("List of available providers: %s", self.provider_list)
        
        return self.provider_list
    
    def create_provider(self, name, config):
        log.info("Creating '%s' provider", name)
        provider_class = self.get_provider_class(name)
        
        if provider_class is None:
            log.exception("A provider with the name %s could not "
                          "be found", name)
            raise NotImplementedError(
                'A provider with name {$name} could not be found'
            )
        log.debug("Created '%s' provider", name)
        return provider_class(config)
    
    def get_provider_class(self, name):
        log.debug("Returning a class for the %s provider", name)
        impl = self.list_providers().get(name)
        if impl:
            log.debug("Returning provider class for %s", name)
            return impl["class"]
        else:
            log.debug("Provider with the name: %s not found", name)
            return None
        
    def get_all_provider_classes(self, ignore_mocks=False):
        all_providers = []
        for impl in self.list_providers().values():
            if ignore_mocks:
                if not issubclass(impl["class"], TestMockHelperMixin):
                    all_providers.append(impl["class"])
            else:
                all_providers.append(impl["class"])
        log.info("List of provider classes: %s", all_providers)
        return all_providers