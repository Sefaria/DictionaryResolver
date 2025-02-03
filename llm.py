from langchain_anthropic import ChatAnthropic
from langchain_core.rate_limiters import InMemoryRateLimiter


rate_limiter = InMemoryRateLimiter(
    requests_per_second=1,
    check_every_n_seconds=0.25,
    max_bucket_size = 400000 / 60
)

model = ChatAnthropic(model="claude-3-5-sonnet-20241022", rate_limiter=rate_limiter)
