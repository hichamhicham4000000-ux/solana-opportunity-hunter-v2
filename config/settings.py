from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv


load_dotenv()


_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class Settings(BaseSettings):
    """
    Centralized configuration for the Solana Opportunity Hunter.

    Discord is used as the user interface.
    The analysis engine remains independent from Discord.
    """

    # ------------------------------------------------------------------
    # Discord
    # ------------------------------------------------------------------

    discord_bot_token: str = ""

    # Optional Discord server (guild) ID.
    # Useful later for guild-specific commands and synchronization.
    discord_guild_id: str = ""

    # Optional Discord channel for automatic opportunity reports.
    discord_channel_id: str = ""

    # Comma-separated Discord user IDs with administrator privileges.
    admin_ids: str = ""

    # ------------------------------------------------------------------
    # Solana / Helius
    # ------------------------------------------------------------------

    helius_api_key: str = ""

    solana_rpc_url: str = "https://api.mainnet-beta.solana.com"

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------

    database_url: str = "sqlite+aiosqlite:///rugscore.db"

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    cache_ttl_seconds: int = Field(default=60, ge=0)

    # ------------------------------------------------------------------
    # Analysis limits
    # ------------------------------------------------------------------

    analysis_timeout_seconds: int = Field(default=20, gt=0)

    max_concurrent_analyses: int = Field(
        default=20,
        gt=0,
        le=100,
    )

    # ------------------------------------------------------------------
    # Opportunity Hunter
    # ------------------------------------------------------------------

    # Maximum number of tokens considered in one discovery cycle.
    max_discovery_tokens: int = Field(
        default=100,
        gt=0,
        le=1000,
    )

    # Minimum overall score before a token can enter the opportunity
    # ranking layer.
    opportunity_min_score: int = Field(
        default=60,
        ge=0,
        le=100,
    )

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------

    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        value = value.upper()

        if value not in _VALID_LOG_LEVELS:
            raise ValueError(
                f"log_level must be one of {_VALID_LOG_LEVELS}, got '{value}'"
            )

        return value

    # ------------------------------------------------------------------
    # Admin helpers
    # ------------------------------------------------------------------

    @property
    def admin_id_set(self) -> set[int]:
        """
        Convert comma-separated ADMIN_IDS into a set of Discord user IDs.
        """

        if not self.admin_ids:
            return set()

        ids: set[int] = set()

        for part in self.admin_ids.split(","):
            part = part.strip()

            if part.isdigit():
                ids.add(int(part))

        return ids

    def is_admin(self, discord_user_id: int) -> bool:
        """
        Check whether a Discord user has administrator privileges.
        """

        return discord_user_id in self.admin_id_set

    # ------------------------------------------------------------------
    # Helius helpers
    # ------------------------------------------------------------------

    @property
    def helius_rpc_url(self) -> str:
        """
        Build the Helius RPC endpoint when an API key is available.
        """

        if not self.helius_api_key:
            return self.solana_rpc_url

        return (
            f"https://mainnet.helius-rpc.com/"
            f"?api-key={self.helius_api_key}"
        )

    @property
    def helius_api_url(self) -> str:
        """
        Helius REST API base URL.
        """

        return "https://api.helius.xyz/v0"

    # ------------------------------------------------------------------
    # Startup validation
    # ------------------------------------------------------------------

    def validate_startup(self) -> list[str]:
        """
        Return non-fatal startup warnings.

        The application should not crash merely because optional
        integrations such as Helius are not configured.
        """

        warnings: list[str] = []

        if not self.discord_bot_token:
            warnings.append(
                "DISCORD_BOT_TOKEN not set"
            )

        if not self.helius_api_key:
            warnings.append(
                "HELIUS_API_KEY not set -- some features limited"
            )

        if not self.admin_id_set:
            warnings.append(
                "ADMIN_IDS not configured -- admin commands inaccessible"
            )

        return warnings


class Config:
    env_file = ".env"
    env_file_encoding = "utf-8"


settings = Settings()
