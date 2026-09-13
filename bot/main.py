"""
Discord entry point for Solana Opportunity Hunter.

The Discord layer is intentionally kept separate from the analysis engine.
The analysis engine remains responsible for Solana/token intelligence.
"""

import asyncio
import logging
import os
import sys
import time
from typing import Any

import discord
from discord import app_commands
from discord.ext import commands

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Internal modules
# ---------------------------------------------------------------------------

from config.settings import settings
from database.db import init_db
from data.cache import cache
from smart_money.wallet_list import load_from_db
from analysis.engine import analyze_token, quick_score


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure application logging."""

    log_level = getattr(
        logging,
        settings.log_level.upper(),
        logging.INFO,
    )

    logging.basicConfig(
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        level=log_level,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("discord").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Discord bot
# ---------------------------------------------------------------------------


class OpportunityHunterBot(commands.Bot):
    """Main Discord application."""

    def __init__(self) -> None:
        intents = discord.Intents.default()

        super().__init__(
            command_prefix="!",
            intents=intents,
            help_command=None,
        )

        self.start_time = time.time()
        self._synced = False

    async def setup_hook(self) -> None:
        """Initialize services and synchronize slash commands."""

        logger.info("Initializing database...")
        await init_db()

        logger.info("Loading smart-money wallets...")
        try:
            await load_from_db()
        except Exception as exc:
            logger.warning(
                "Could not load smart-money wallets: %s",
                exc,
            )

        logger.info("Starting cache cleanup...")
        try:
            cache.start_cleanup_task()
        except Exception as exc:
            logger.warning(
                "Could not start cache cleanup: %s",
                exc,
            )

        # Sync global slash commands.
        try:
            synced = await self.tree.sync()

            logger.info(
                "Discord slash commands synchronized: %s",
                len(synced),
            )

            self._synced = True

        except Exception as exc:
            logger.exception(
                "Failed to synchronize Discord commands: %s",
                exc,
            )

    async def close(self) -> None:
        """Clean up application resources before Discord shutdown."""

        logger.info("Shutting down Opportunity Hunter...")

        try:
            cache.stop_cleanup_task()
        except Exception as exc:
            logger.warning(
                "Cache cleanup shutdown warning: %s",
                exc,
            )

        # Close HTTP clients used by the analysis layer.
        modules_to_close = (
            "data.solana_rpc",
            "data.helius_client",
            "data.dexscreener",
            "data.jupiter",
            "data.social_checker",
            "data.rugcheck_client",
        )

        for module_name in modules_to_close:
            try:
                module = __import__(
                    module_name,
                    fromlist=["*"],
                )

                close_function = getattr(
                    module,
                    "close",
                    None,
                )

                if close_function:
                    result = close_function()

                    if asyncio.iscoroutine(result):
                        await result

            except ModuleNotFoundError:
                # Optional module.
                continue

            except Exception as exc:
                logger.warning(
                    "Could not close %s: %s",
                    module_name,
                    exc,
                )

        await super().close()


bot = OpportunityHunterBot()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_valid_solana_address(value: str) -> bool:
    """Validate a Solana public key."""

    try:
        from solders.pubkey import Pubkey

        Pubkey.from_string(value)
        return True

    except Exception:
        return False


def shorten_address(address: str) -> str:
    """Shorten a Solana address for Discord display."""

    if len(address) <= 12:
        return address

    return f"{address[:6]}...{address[-6:]}"


def format_usd(value: Any) -> str:
    """Format USD values safely."""

    if value is None:
        return "غير متوفر"

    try:
        number = float(value)

        if number >= 1_000_000_000:
            return f"${number / 1_000_000_000:.2f}B"

        if number >= 1_000_000:
            return f"${number / 1_000_000:.2f}M"

        if number >= 1_000:
            return f"${number / 1_000:.2f}K"

        if number < 0.01:
            return f"${number:.8f}"

        return f"${number:.4f}"

    except (TypeError, ValueError):
        return "غير متوفر"


def risk_color(score: float) -> discord.Color:
    """Return a Discord embed color based on the score."""

    if score >= 80:
        return discord.Color.green()

    if score >= 60:
        return discord.Color.blue()

    if score >= 40:
        return discord.Color.gold()

    return discord.Color.red()


CRITERIA_ARABIC = {
    "contract": "أمان العقد",
    "liquidity": "جودة السيولة",
    "holders": "توزيع الحائزين",
    "dev_wallet": "محفظة المطور",
    "volume": "جودة الحجم",
    "social": "المصداقية الاجتماعية",
    "metadata": "جودة البيانات",
    "smart_money": "Smart Money",
}


def build_analysis_embed(
    result: dict,
    *,
    quick: bool = False,
) -> discord.Embed:
    """Convert an analysis result into a Discord embed."""

    score = float(result.get("total_score", 0) or 0)

    symbol = result.get("token_symbol") or "???"
    name = result.get("token_name") or "Unknown"
    mint = result.get("token_mint") or ""

    risk_label = result.get(
        "risk_label",
        "غير محدد",
    )

    risk_desc = result.get(
        "risk_desc",
        "",
    )

    risk_emoji = result.get(
        "risk_emoji",
        "⚠️",
    )

    title_prefix = "⚡ فحص سريع" if quick else "🔎 تحليل Solana"

    embed = discord.Embed(
        title=f"{title_prefix} — ${symbol}",
        description=(
            f"**{name}**\n"
            f"{risk_emoji} **{risk_label}**\n"
            f"{risk_desc}"
        ),
        color=risk_color(score),
    )

    embed.add_field(
        name="📊 النتيجة",
        value=f"**{score:.1f}/100**",
        inline=True,
    )

    embed.add_field(
        name="💰 السعر",
        value=format_usd(
            result.get("price_usd")
        ),
        inline=True,
    )

    embed.add_field(
        name="💧 السيولة",
        value=format_usd(
            result.get("liquidity_usd")
        ),
        inline=True,
    )

    embed.add_field(
        name="🏦 Market Cap",
        value=format_usd(
            result.get("market_cap")
        ),
        inline=True,
    )

    if result.get("volume_24h") is not None:
        embed.add_field(
            name="📈 حجم 24 ساعة",
            value=format_usd(
                result.get("volume_24h")
            ),
            inline=True,
        )

    analysis_time = result.get("analysis_time")

    if analysis_time is not None:
        embed.add_field(
            name="⏱️ زمن التحليل",
            value=f"{analysis_time}s",
            inline=True,
        )

    # Full analysis criteria.
    criteria = result.get("criteria")

    if isinstance(criteria, dict) and criteria:
        lines = []

        for key, criterion in criteria.items():
            label = CRITERIA_ARABIC.get(
                key,
                key.replace("_", " ").title(),
            )

            criterion_score = getattr(
                criterion,
                "score",
                None,
            )

            if criterion_score is None:
                criterion_score = 0

            lines.append(
                f"**{label}:** {float(criterion_score):.0f}/100"
            )

        if lines:
            embed.add_field(
                name="🧠 معايير التحليل",
                value="\n".join(lines),
                inline=False,
            )

    # Risk flags.
    flags = result.get("all_flags") or []

    if flags:
        flag_lines = []

        for flag in flags[:8]:
            flag_lines.append(
                f"• {flag}"
            )

        embed.add_field(
            name="⚠️ إشارات المخاطر",
            value="\n".join(flag_lines),
            inline=False,
        )

    data_quality = result.get("data_quality")

    if data_quality is not None:
        embed.add_field(
            name="🧪 جودة البيانات",
            value=f"{data_quality}%",
            inline=True,
        )

    if mint:
        embed.add_field(
            name="🪙 Mint",
            value=f"`{shorten_address(mint)}`",
            inline=False,
        )

        embed.add_field(
            name="🔗 روابط",
            value=(
                f"[Solscan](https://solscan.io/token/{mint}) • "
                f"[DexScreener]"
                f"(https://dexscreener.com/solana/{mint})"
            ),
            inline=False,
        )

    embed.set_footer(
        text=(
            "Solana Opportunity Hunter • "
            "تحليل آلي — ليس ضمانًا للربح"
        )
    )

    return embed


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------


@bot.tree.command(
    name="help",
    description="عرض أوامر Solana Opportunity Hunter",
)
async def help_command(
    interaction: discord.Interaction,
) -> None:
    """Show available commands."""

    embed = discord.Embed(
        title="🤖 Solana Opportunity Hunter",
        description=(
            "نظام تحليل واكتشاف فرص على شبكة Solana.\n\n"
            "**الأوامر الحالية:**\n"
            "🔎 `/scan` — تحليل عميق للتوكن\n"
            "⚡ `/quick` — فحص سريع\n"
            "🆔 `/myid` — عرض Discord User ID\n"
            "❤️ `/health` — حالة النظام\n\n"
            "⚠️ النتائج تحليلية وليست ضمانًا للربح."
        ),
        color=discord.Color.blurple(),
    )

    await interaction.response.send_message(
        embed=embed
    )


@bot.tree.command(
    name="scan",
    description="تحليل عميق لتوكن Solana",
)
@app_commands.describe(
    mint="عنوان Mint الكامل للتوكن على Solana",
)
async def scan_command(
    interaction: discord.Interaction,
    mint: str,
) -> None:
    """Run a complete token analysis."""

    mint = mint.strip()

    if not is_valid_solana_address(mint):
        await interaction.response.send_message(
            "❌ عنوان Mint غير صالح على شبكة Solana.",
            ephemeral=True,
        )
        return

    await interaction.response.defer(
        thinking=True
    )

    try:
        result = await analyze_token(mint)

        embed = build_analysis_embed(
            result,
            quick=False,
        )

        await interaction.followup.send(
            embed=embed
        )

    except Exception as exc:
        logger.exception(
            "Full analysis failed for %s: %s",
            mint,
            exc,
        )

        await interaction.followup.send(
            "❌ حدث خطأ أثناء تحليل التوكن. "
            "تحقق من العنوان وحاول مرة أخرى."
        )


@bot.tree.command(
    name="quick",
    description="فحص سريع لأهم مخاطر توكن Solana",
)
@app_commands.describe(
    mint="عنوان Mint الكامل للتوكن على Solana",
)
async def quick_command(
    interaction: discord.Interaction,
    mint: str,
) -> None:
    """Run a quick token analysis."""

    mint = mint.strip()

    if not is_valid_solana_address(mint):
        await interaction.response.send_message(
            "❌ عنوان Mint غير صالح على شبكة Solana.",
            ephemeral=True,
        )
        return

    await interaction.response.defer(
        thinking=True
    )

    try:
        result = await quick_score(mint)

        embed = build_analysis_embed(
            result,
            quick=True,
        )

        await interaction.followup.send(
            embed=embed
        )

    except Exception as exc:
        logger.exception(
            "Quick analysis failed for %s: %s",
            mint,
            exc,
        )

        await interaction.followup.send(
            "❌ حدث خطأ أثناء الفحص السريع."
        )


@bot.tree.command(
    name="myid",
    description="عرض Discord User ID الخاص بك",
)
async def myid_command(
    interaction: discord.Interaction,
) -> None:
    """Return the current Discord user ID."""

    await interaction.response.send_message(
        f"🆔 Discord User ID:\n`{interaction.user.id}`",
        ephemeral=True,
    )


@bot.tree.command(
    name="health",
    description="عرض حالة البوت ومحرك التحليل",
)
async def health_command(
    interaction: discord.Interaction,
) -> None:
    """Return basic system health."""

    uptime = int(
        time.time() - bot.start_time
    )

    hours, remainder = divmod(
        uptime,
        3600,
    )

    minutes, seconds = divmod(
        remainder,
        60,
    )

    helius_status = (
        "🟢 متصل"
        if settings.helius_api_key
        else "🟡 غير مضبوط"
    )

    embed = discord.Embed(
        title="❤️ حالة النظام",
        color=discord.Color.green(),
    )

    embed.add_field(
        name="Discord",
        value="🟢 متصل",
        inline=True,
    )

    embed.add_field(
        name="Helius",
        value=helius_status,
        inline=True,
    )

    embed.add_field(
        name="Analysis Engine",
        value="🟢 جاهز",
        inline=True,
    )

    embed.add_field(
        name="Uptime",
        value=(
            f"{hours}h "
            f"{minutes}m "
            f"{seconds}s"
        ),
        inline=True,
    )

    embed.set_footer(
        text="Solana Opportunity Hunter"
    )

    await interaction.response.send_message(
        embed=embed,
        ephemeral=True,
    )


# ---------------------------------------------------------------------------
# Discord events
# ---------------------------------------------------------------------------


@bot.event
async def on_ready() -> None:
    """Called when Discord connection is ready."""

    if bot.user is None:
        return

    logger.info(
        "Discord bot connected as %s (%s)",
        bot.user,
        bot.user.id,
    )

    logger.info(
        "Connected to %s Discord guild(s).",
        len(bot.guilds),
    )

    logger.info(
        "Slash commands synchronized: %s",
        bot._synced,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    """Start the Discord bot."""

    setup_logging()

    logger.info(
        "Starting Solana Opportunity Hunter..."
    )

    # Startup warnings are intentionally non-fatal.
    for warning in settings.validate_startup():
        logger.warning(warning)

    if not settings.discord_bot_token:
        logger.error(
            "DISCORD_BOT_TOKEN is not configured."
        )
        return

    try:
        async with bot:
            await bot.start(
                settings.discord_bot_token
            )

    except discord.LoginFailure:
        logger.error(
            "Discord login failed. "
            "Check the Discord bot token."
        )

    except Exception:
        logger.exception(
            "Discord bot stopped unexpectedly."
        )


if __name__ == "__main__":
    asyncio.run(main())
