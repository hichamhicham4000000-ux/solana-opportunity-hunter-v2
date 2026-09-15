"""
Discord entry point for Solana Opportunity Hunter.

Discord is only the presentation/control layer.

The analytical pipeline is responsible for:
    Discovery
    Ranking
    Security Gate
    Deep Intelligence
    Smart Money
    Market Intelligence
    Manipulation Risk
    Opportunity Decision

No real-money trading is executed by this bot.
"""

from __future__ import annotations

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
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT,
    )


# ---------------------------------------------------------------------------
# Internal modules
# ---------------------------------------------------------------------------

from config.settings import settings
from database.db import init_db
from data.cache import cache
from smart_money.wallet_list import load_from_db

from analysis.engine import (
    quick_score,
)

from analysis.opportunity_pipeline import (
    opportunity_pipeline,
    OpportunityPipelineResult,
)

from analysis.deep_intelligence import (
    DeepIntelligenceResult,
)


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
        format=(
            "%(asctime)s "
            "[%(name)s] "
            "%(levelname)s: "
            "%(message)s"
        ),
        level=log_level,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
    )

    logging.getLogger(
        "httpx"
    ).setLevel(
        logging.WARNING
    )

    logging.getLogger(
        "httpcore"
    ).setLevel(
        logging.WARNING
    )

    logging.getLogger(
        "discord"
    ).setLevel(
        logging.WARNING
    )


# ---------------------------------------------------------------------------
# Discord bot
# ---------------------------------------------------------------------------


class OpportunityHunterBot(
    commands.Bot
):
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

    async def setup_hook(
        self,
    ) -> None:
        """Initialize services and synchronize slash commands."""

        logger.info(
            "Initializing database..."
        )

        await init_db()

        logger.info(
            "Loading smart-money wallets..."
        )

        try:

            await load_from_db()

        except Exception as exc:

            logger.warning(
                "Could not load smart-money wallets: %s",
                exc,
            )

        logger.info(
            "Starting cache cleanup..."
        )

        try:

            cache.start_cleanup_task()

        except Exception as exc:

            logger.warning(
                "Could not start cache cleanup: %s",
                exc,
            )

        # Synchronize slash commands.
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

    async def close(
        self,
    ) -> None:
        """Clean up application resources."""

        logger.info(
            "Shutting down Opportunity Hunter..."
        )

        try:

            cache.stop_cleanup_task()

        except Exception as exc:

            logger.warning(
                "Cache cleanup shutdown warning: %s",
                exc,
            )

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

                    if asyncio.iscoroutine(
                        result
                    ):
                        await result

            except ModuleNotFoundError:

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


def is_valid_solana_address(
    value: str,
) -> bool:
    """Validate a Solana public key."""

    try:

        from solders.pubkey import Pubkey

        Pubkey.from_string(
            value
        )

        return True

    except Exception:

        return False


def shorten_address(
    address: str,
) -> str:
    """Shorten a Solana address."""

    if len(address) <= 12:
        return address

    return (
        f"{address[:6]}"
        f"..."
        f"{address[-6:]}"
    )


def format_usd(
    value: Any,
) -> str:
    """Format USD values safely."""

    if value is None:
        return "غير متوفر"

    try:

        number = float(value)

        if number >= 1_000_000_000:
            return (
                f"${number / 1_000_000_000:.2f}B"
            )

        if number >= 1_000_000:
            return (
                f"${number / 1_000_000:.2f}M"
            )

        if number >= 1_000:
            return (
                f"${number / 1_000:.2f}K"
            )

        if number < 0.01:
            return f"${number:.8f}"

        return f"${number:.4f}"

    except (
        TypeError,
        ValueError,
    ):

        return "غير متوفر"


def format_percent(
    value: Any,
) -> str:
    """Format percentage values."""

    try:

        return f"{float(value):.1f}%"

    except (
        TypeError,
        ValueError,
    ):

        return "غير متوفر"


def score_bar(
    score: float,
) -> str:
    """
    Small visual score indicator for Discord.

    This is only presentation and does not change scoring.
    """

    score = max(
        0.0,
        min(
            float(score),
            100.0,
        ),
    )

    filled = int(
        round(score / 10)
    )

    filled = max(
        0,
        min(
            filled,
            10,
        ),
    )

    return (
        "🟩" * filled
        + "⬜" * (10 - filled)
    )


def score_color(
    score: float,
) -> discord.Color:
    """Return a Discord color based on a score."""

    if score >= 82:
        return discord.Color.green()

    if score >= 68:
        return discord.Color.blue()

    if score >= 50:
        return discord.Color.gold()

    return discord.Color.red()


def decision_arabic(
    decision: str,
) -> tuple[str, str]:
    """Translate analytical decision to Arabic."""

    mapping = {
        # Current Deep Intelligence decisions.
        "BUY": (
            "🟢 شراء / فرصة تداول",
            "توجد مجموعة إشارات متوافقة تسمح بدراسة دخول قصير الأجل",
        ),

        "WAIT": (
            "🟡 انتظار",
            "الإشارات الحالية غير كافية لتأكيد دخول قصير الأجل",
        ),

        "AVOID": (
            "🔴 تجنب",
            "المخاطر أو شروط التحليل تمنع الدخول حاليًا",
        ),

        # Backward-compatible aliases.
        "STRONG OPPORTUNITY": (
            "🟢 شراء / فرصة قوية",
            "استحقاق مرتفع للدراسة والمتابعة",
        ),

        "OPPORTUNITY": (
            "🟢 شراء / فرصة",
            "توجد مجموعة إشارات إيجابية تستحق المتابعة",
        ),

        "WATCH": (
            "🟡 انتظار",
            "الإشارات غير كافية لاتخاذ موقف إيجابي قوي",
        ),

        "SPECULATIVE": (
            "🟡 انتظار",
            "المخاطر أو عدم اليقين مرتفعان",
        ),

        "INVALIDATED": (
            "⛔ ملغاة",
            "لم تعد الفرضية التحليلية صالحة",
        ),
    }

    return mapping.get(
        str(decision).upper(),
        (
            str(decision),
            "لا يوجد وصف متاح",
        ),
    )


def trade_setup_arabic(setup: str) -> str:
    """Translate the short-term trading setup to Arabic."""
    mapping = {
        "BREAKOUT_MOMENTUM": "اختراق + زخم",
        "MOMENTUM": "زخم صاعد",
        "SMART_MONEY_FLOW": "تدفق Smart Money",
        "BUY_PRESSURE": "ضغط شراء",
        "POSITIVE_MARKET_STRUCTURE": "هيكل سوق إيجابي",
        "HIGH_RISK": "مخاطرة مرتفعة",
        "NO_CLEAR_SETUP": "لا توجد إشارة دخول واضحة",
        "SECURITY_BLOCKED": "محجوب أمنيًا",
    }
    return mapping.get(
        str(setup).upper(),
        str(setup),
    )


def risk_color(
    score: float,
) -> discord.Color:
    """
    Security/risk presentation color.

    Higher security score is better.
    """

    if score >= 80:
        return discord.Color.green()

    if score >= 60:
        return discord.Color.blue()

    if score >= 40:
        return discord.Color.gold()

    return discord.Color.red()


# ---------------------------------------------------------------------------
# Deep Opportunity Report
# ---------------------------------------------------------------------------


def build_opportunity_embed(
    result: DeepIntelligenceResult,
) -> discord.Embed:
    """
    Build the main Arabic Opportunity Hunter report.

    This is presentation only.
    The underlying result remains the source of truth.
    """

    opportunity = float(
        result.opportunity_score
    )

    confidence = float(
        result.confidence
    )

    decision_title, decision_description = (
        decision_arabic(
            result.decision
        )
    )

    embed = discord.Embed(
        title=(
            f"{decision_title} — "
            f"${result.token_symbol}"
        ),
        description=(
            f"**{result.token_name}**\n"
            f"{decision_description}\n\n"
            f"{score_bar(opportunity)}\n"
            f"🎯 **Opportunity Score:** "
            f"`{opportunity:.1f}/100`"
        ),
        color=score_color(
            opportunity
        ),
    )

    # ---------------------------------------------------------------
    # Core decision
    # ---------------------------------------------------------------

    embed.add_field(
        name="🎯 القرار التحليلي",
        value=(
            f"**{decision_title}**\n"
            f"الكود: `{result.decision}`\n"
            f"ثقة التحليل: `{confidence:.1f}%`"
        ),
        inline=True,
    )

    embed.add_field(
        name="⏱️ خطة التداول القصير",
        value=(
            f"الإجراء: **{result.trade_action}**\n"
            f"الإعداد: `{trade_setup_arabic(result.trade_setup)}`\n"
            f"الأفق: `{result.trade_horizon}`"
        ),
        inline=True,
    )

    embed.add_field(
        name="🛡️ الأمان",
        value=(
            f"`{result.security_score:.1f}/100`\n"
            f"{score_bar(result.security_score)}"
        ),
        inline=True,
    )

    embed.add_field(
        name="📊 الترتيب",
        value=(
            f"`{result.rank_score:.1f}/100`\n"
            f"{score_bar(result.rank_score)}"
        ),
        inline=True,
    )

    # ---------------------------------------------------------------
    # Market intelligence
    # ---------------------------------------------------------------

    embed.add_field(
        name="📈 ذكاء السوق",
        value=(
            f"السوق: `{result.market_score:.1f}/100`\n"
            f"الزخم: `{result.momentum_score:.1f}`\n"
            f"الحجم: `{result.volume_score:.1f}`\n"
            f"الضغط: `{result.pressure_score:.1f}`"
        ),
        inline=True,
    )

    embed.add_field(
        name="💧 السيولة",
        value=(
            f"`{result.liquidity_score:.1f}/100`\n"
            f"الهيكل: `{result.structure_score:.1f}`\n"
            f"Breakout: `{result.breakout_score:.1f}`"
        ),
        inline=True,
    )

    embed.add_field(
        name="🐋 Smart Money",
        value=(
            f"Score: `{result.smart_money_score:.1f}/100`\n"
            f"State: `{result.smart_money_state}`\n"
            f"Confidence: "
            f"`{result.smart_money_confidence:.1f}%`"
        ),
        inline=True,
    )

    # ---------------------------------------------------------------
    # Manipulation
    # ---------------------------------------------------------------

    manipulation_level = (
        result.manipulation_risk_level
    )

    manipulation_emoji = "🟢"

    if manipulation_level == "MEDIUM":
        manipulation_emoji = "🟡"

    elif manipulation_level == "HIGH":
        manipulation_emoji = "🟠"

    elif manipulation_level == "CRITICAL":
        manipulation_emoji = "🔴"

    embed.add_field(
        name="⚠️ مخاطر التلاعب",
        value=(
            f"{manipulation_emoji} "
            f"`{manipulation_level}`\n"
            f"Score: "
            f"`{result.manipulation_score:.1f}/100`"
        ),
        inline=True,
    )

    embed.add_field(
        name="🧪 جودة البيانات",
        value=(
            f"`{result.data_quality:.1f}/100`"
        ),
        inline=True,
    )

    embed.add_field(
        name="🔎 Discovery",
        value=(
            f"`{result.discovery_score:.1f}/100`"
        ),
        inline=True,
    )

    # ---------------------------------------------------------------
    # Reasons
    # ---------------------------------------------------------------

    reasons = result.reasons or []

    if reasons:

        lines = [
            f"• {reason}"
            for reason in reasons[:8]
        ]

        embed.add_field(
            name="✅ لماذا ظهرت هذه النتيجة؟",
            value="\n".join(lines),
            inline=False,
        )

    # ---------------------------------------------------------------
    # Warnings
    # ---------------------------------------------------------------

    warnings = result.warnings or []

    if warnings:

        lines = [
            f"• {warning}"
            for warning in warnings[:8]
        ]

        embed.add_field(
            name="⚠️ التحذيرات",
            value="\n".join(lines),
            inline=False,
        )

    # ---------------------------------------------------------------
    # Blockers
    # ---------------------------------------------------------------

    blockers = result.blockers or []

    if blockers:

        lines = [
            f"• {blocker}"
            for blocker in blockers[:6]
        ]

        embed.add_field(
            name="🚫 أسباب الحجب",
            value="\n".join(lines),
            inline=False,
        )

    # ---------------------------------------------------------------
    # Token links
    # ---------------------------------------------------------------

    mint = result.mint

    if mint:

        embed.add_field(
            name="🪙 Mint",
            value=(
                f"`{shorten_address(mint)}`"
            ),
            inline=False,
        )

        embed.add_field(
            name="🔗 روابط",
            value=(
                f"[Solscan]"
                f"(https://solscan.io/token/{mint})"
                f" • "
                f"[DexScreener]"
                f"(https://dexscreener.com/solana/{mint})"
            ),
            inline=False,
        )

    embed.set_footer(
        text=(
            "Solana Opportunity Hunter • "
            "تحليل احتمالي/استدلالي — "
            "ليس ضمانًا للربح"
        )
    )

    return embed


# ---------------------------------------------------------------------------
# Security rejection report
# ---------------------------------------------------------------------------


def build_avoid_embed(
    result: DeepIntelligenceResult,
) -> discord.Embed:
    """Build an accurate report for any AVOID result.

    AVOID can come from the Security Gate or from Deep Intelligence
    after the security screen, so the report must distinguish both cases.
    """

    security_blocked = (
        result.trade_setup == "SECURITY_BLOCKED"
    )

    if security_blocked:
        title = f"🔴 تجنب — ${result.token_symbol}"
        description = (
            f"**{result.token_name}**\n\n"
            "تم رفض التوكن بواسطة **Security Gate** "
            "ولم يسمح النظام بمتابعة التحليل العميق."
        )
    else:
        title = f"🔴 تجنب — ${result.token_symbol}"
        description = (
            f"**{result.token_name}**\n\n"
            "التحليل العميق لا يرى دخولًا مناسبًا "
            "للتداول قصير الأجل في الظروف الحالية."
        )

    embed = discord.Embed(
        title=title,
        description=description,
        color=discord.Color.red(),
    )

    embed.add_field(
        name="🎯 القرار",
        value=(
            f"**تجنب**\n"
            f"الكود: `{result.decision}`\n"
            f"الثقة: `{result.confidence:.1f}%`"
        ),
        inline=True,
    )

    embed.add_field(
        name="🛡️ الأمان",
        value=f"`{result.security_score:.1f}/100`",
        inline=True,
    )

    embed.add_field(
        name="🎯 Opportunity Score",
        value=f"`{result.opportunity_score:.1f}/100`",
        inline=True,
    )

    embed.add_field(
        name="⚠️ مخاطر التلاعب",
        value=(
            f"`{result.manipulation_risk_level}`\n"
            f"Score: `{result.manipulation_score:.1f}/100`"
        ),
        inline=True,
    )

    embed.add_field(
        name="📈 السوق",
        value=(
            f"Market: `{result.market_score:.1f}`\n"
            f"Momentum: `{result.momentum_score:.1f}`\n"
            f"Pressure: `{result.pressure_score:.1f}`"
        ),
        inline=True,
    )

    embed.add_field(
        name="⏱️ التداول القصير",
        value=(
            f"الإعداد: `{trade_setup_arabic(result.trade_setup)}`\n"
            f"الأفق: `{result.trade_horizon}`"
        ),
        inline=True,
    )

    blockers = result.blockers or []
    if blockers:
        embed.add_field(
            name="🚫 أسباب الحجب/المنع",
            value="\n".join(
                f"• {item}"
                for item in blockers[:8]
            ),
            inline=False,
        )

    reasons = result.reasons or []
    if reasons:
        embed.add_field(
            name="📌 الأسباب الرئيسية",
            value="\n".join(
                f"• {item}"
                for item in reasons[:8]
            ),
            inline=False,
        )

    warnings = result.warnings or []
    if warnings:
        embed.add_field(
            name="⚠️ التحذيرات",
            value="\n".join(
                f"• {item}"
                for item in warnings[:8]
            ),
            inline=False,
        )

    mint = result.mint
    if mint:
        embed.add_field(
            name="🪙 Mint",
            value=f"`{shorten_address(mint)}`",
            inline=False,
        )

        embed.add_field(
            name="🔗 روابط",
            value=(
                f"[Solscan](https://solscan.io/token/{mint})"
                f" • "
                f"[DexScreener](https://dexscreener.com/solana/{mint})"
            ),
            inline=False,
        )

    embed.set_footer(
        text=(
            "AVOID = لا يوجد دخول مناسب حاليًا. "
            "لا يوجد تنفيذ تلقائي للصفقات."
        )
    )

    return embed


# Backward-compatible function name for any existing imports.
def build_security_rejection_embed(
    result: DeepIntelligenceResult,
) -> discord.Embed:
    return build_avoid_embed(result)


# ---------------------------------------------------------------------------
# Pipeline summary
# ---------------------------------------------------------------------------


def build_pipeline_summary_embed(
    result: OpportunityPipelineResult,
) -> discord.Embed:
    """Build a summary of an opportunity-hunting cycle."""

    embed = discord.Embed(
        title="🎯 Opportunity Hunter",
        description=(
            "ملخص دورة البحث والتحليل الحالية."
        ),
        color=discord.Color.blurple(),
    )

    embed.add_field(
        name="🔎 المكتشف",
        value=str(
            len(result.candidates)
        ),
        inline=True,
    )

    embed.add_field(
        name="📊 بعد الترتيب",
        value=str(
            len(result.ranked)
        ),
        inline=True,
    )

    passed = sum(
        1
        for _ranked, decision
        in result.security_results
        if decision.passed
    )

    embed.add_field(
        name="🛡️ اجتاز الأمان",
        value=str(passed),
        inline=True,
    )

    embed.add_field(
        name="🧠 Deep Intelligence",
        value=str(
            len(result.deep_results)
        ),
        inline=True,
    )

    embed.add_field(
        name="💎 فرص قابلة للدراسة",
        value=str(
            len(result.actionable)
        ),
        inline=True,
    )

    embed.add_field(
        name="⏱️ الزمن",
        value=(
            f"{result.elapsed_seconds:.2f}s"
        ),
        inline=True,
    )

    if result.actionable:

        lines = []

        for item in result.actionable[:10]:

            lines.append(
                f"**${item.token_symbol}** — "
                f"`{item.opportunity_score:.1f}` "
                f"— `{item.decision}`"
            )

        embed.add_field(
            name="🏆 أفضل النتائج",
            value="\n".join(lines),
            inline=False,
        )

    else:

        embed.add_field(
            name="📭 النتيجة",
            value=(
                "لم يتم العثور على فرصة "
                "تستوفي شروط التحليل الحالية."
            ),
            inline=False,
        )

    embed.set_footer(
        text=(
            "نتائج تحليلية — "
            "لا يوجد تنفيذ تلقائي للصفقات."
        )
    )

    return embed


# ---------------------------------------------------------------------------
# Slash command: help
# ---------------------------------------------------------------------------


@bot.tree.command(
    name="help",
    description=(
        "عرض أوامر Solana Opportunity Hunter"
    ),
)
async def help_command(
    interaction: discord.Interaction,
) -> None:
    """Show available commands."""

    embed = discord.Embed(
        title="🤖 Solana Opportunity Hunter",
        description=(
            "نظام تحليل واكتشاف فرص على شبكة Solana.\n\n"

            "**الأوامر:**\n"

            "🔎 `/scan` — تحليل عميق + قرار BUY / WAIT / AVOID\n"
            "⚡ `/quick` — فحص سريع للمخاطر الأساسية\n"
            "🆔 `/myid` — Discord User ID\n"
            "❤️ `/health` — حالة النظام\n\n"

            "**مراحل التحليل:**\n"
            "Discovery → Ranking → Security → "
            "Deep Intelligence → Market → Manipulation → Decision\n\n"

            "⚠️ النتائج تحليلية وليست ضمانًا للربح."
        ),
        color=discord.Color.blurple(),
    )

    await interaction.response.send_message(
        embed=embed
    )


# ---------------------------------------------------------------------------
# Slash command: scan
# ---------------------------------------------------------------------------


@bot.tree.command(
    name="scan",
    description=(
        "تحليل عميق لتوكن Solana"
    ),
)
@app_commands.describe(
    mint=(
        "عنوان Mint الكامل للتوكن على Solana"
    ),
)
async def scan_command(
    interaction: discord.Interaction,
    mint: str,
) -> None:
    """
    Run the complete Opportunity Hunter analysis
    for one token.
    """

    mint = mint.strip()

    if not is_valid_solana_address(
        mint
    ):

        await interaction.response.send_message(
            "❌ عنوان Mint غير صالح على شبكة Solana.",
            ephemeral=True,
        )

        return

    await interaction.response.defer(
        thinking=True
    )

    try:

        logger.info(
            "Discord /scan requested for %s",
            mint,
        )

        result = (
            await opportunity_pipeline.analyze_mint(
                mint
            )
        )

        if result.decision == "AVOID":

            embed = build_avoid_embed(result)

        else:

            embed = (
                build_opportunity_embed(
                    result
                )
            )

        await interaction.followup.send(
            embed=embed
        )

    except Exception as exc:

        logger.exception(
            "Opportunity analysis failed for %s: %s",
            mint,
            exc,
        )

        await interaction.followup.send(
            "❌ حدث خطأ أثناء التحليل العميق.\n"
            "تحقق من عنوان Mint وحاول مرة أخرى."
        )


# ---------------------------------------------------------------------------
# Slash command: quick
# ---------------------------------------------------------------------------


@bot.tree.command(
    name="quick",
    description=(
        "فحص سريع لأهم مخاطر توكن Solana"
    ),
)
@app_commands.describe(
    mint=(
        "عنوان Mint الكامل للتوكن على Solana"
    ),
)
async def quick_command(
    interaction: discord.Interaction,
    mint: str,
) -> None:
    """Run a quick token analysis."""

    mint = mint.strip()

    if not is_valid_solana_address(
        mint
    ):

        await interaction.response.send_message(
            "❌ عنوان Mint غير صالح على شبكة Solana.",
            ephemeral=True,
        )

        return

    await interaction.response.defer(
        thinking=True
    )

    try:

        result = await quick_score(
            mint
        )

        score = float(
            result.get(
                "total_score",
                0,
            )
            or 0
        )

        symbol = result.get(
            "token_symbol",
            "???",
        )

        name = result.get(
            "token_name",
            "غير معروف",
        )

        embed = discord.Embed(
            title=(
                f"⚡ فحص سريع — "
                f"${symbol}"
            ),
            description=(
                f"**{name}**\n\n"
                f"النتيجة: "
                f"`{score:.1f}/100`"
            ),
            color=risk_color(
                score
            ),
        )

        embed.add_field(
            name="🛡️ أمان العقد",
            value=(
                f"`{result.get('contract_score', 0)}`"
            ),
            inline=True,
        )

        embed.add_field(
            name="💧 السيولة",
            value=(
                f"`{result.get('liquidity_score', 0)}`"
            ),
            inline=True,
        )

        embed.add_field(
            name="👥 الحائزون",
            value=(
                f"`{result.get('holders_score', 0)}`"
            ),
            inline=True,
        )

        embed.add_field(
            name="🪙 Mint",
            value=(
                f"`{shorten_address(mint)}`"
            ),
            inline=False,
        )

        embed.set_footer(
            text=(
                "الفحص السريع لا يستبدل "
                "التحليل العميق."
            )
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


# ---------------------------------------------------------------------------
# Slash command: myid
# ---------------------------------------------------------------------------


@bot.tree.command(
    name="myid",
    description=(
        "عرض Discord User ID الخاص بك"
    ),
)
async def myid_command(
    interaction: discord.Interaction,
) -> None:
    """Return the current Discord user ID."""

    await interaction.response.send_message(
        (
            "🆔 Discord User ID:\n"
            f"`{interaction.user.id}`"
        ),
        ephemeral=True,
    )


# ---------------------------------------------------------------------------
# Slash command: health
# ---------------------------------------------------------------------------


@bot.tree.command(
    name="health",
    description=(
        "عرض حالة البوت ومحرك التحليل"
    ),
)
async def health_command(
    interaction: discord.Interaction,
) -> None:
    """Return basic system health."""

    uptime = int(
        time.time()
        - bot.start_time
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
        "🟢 مضبوط"
        if settings.helius_api_key
        else "🟡 غير مضبوط"
    )

    embed = discord.Embed(
        title="❤️ حالة النظام",
        color=discord.Color.green(),
    )

    embed.add_field(
        name="Discord",
        value="🟢 يعمل",
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
        name="Opportunity Pipeline",
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

    embed.add_field(
        name="Trading",
        value=(
            "🔒 Paper/Analysis only"
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
# Discord event
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

    for warning in (
        settings.validate_startup()
    ):
        logger.warning(
            warning
        )

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
    asyncio.run(
        main()
)
