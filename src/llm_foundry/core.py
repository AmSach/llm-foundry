from .adapters import EchoBackend, HuggingFacePipelineBackend, OpenAICompatibleBackend, TextBackend, build_backend
from .reasoning import CounterfactualVerifier, MajorityVoteConsensus, ReflectionEngine, ReflectionResult
from .safety import RewardShaper, SafetyLayer, SafetyScore
from .training import CharacterTokenizer, ScratchTrainingConfig, TinyCausalLM, train_from_text
