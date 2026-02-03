from src.custom.credentials.localsettings.langfuseconfig import LangfuseConfig

config = LangfuseConfig()

print("Enabled:", config.enabled)
print("Public Key:", config.public_key)
print("Secret Key:", config.secret_key)
print("Host:", config.host)
print("Flush At:", config.flush_at)
print("Flush Interval:", config.flush_interval)
print("Debug:", config.debug)
