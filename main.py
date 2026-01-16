#!/usr/bin/env python3
"""
🎮 Tic-Tac-Toe Telegram Bot - Production Ready
"""

import asyncio
import sys
import time
import traceback
import logging
import random
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

from pyrogram import Client, filters, enums
from pyrogram.types import (
    Message, InlineKeyboardMarkup, InlineKeyboardButton,
    CallbackQuery, User
)
from pyrogram.errors import MessageNotModified
from pyrogram import idle  # ✅ FIXED IMPORT


from motor.motor_asyncio import AsyncIOMotorClient
from motor.core import AgnosticCollection


# ==================== LOGGING ====================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ==================== CONFIGURATION (HARDCODED - DANGEROUS) ====================

class Settings:
    # ⚠️ HARDCODED CREDENTIALS - EXTREMELY INSECURE
    API_ID = 30779174
    API_HASH = "d5e27c8c4e30129238716a83df45b1f8"
    BOT_TOKEN = "8471916982:AAElBC6sIt-LJTx6cU5a58nA090Rdxo6XxU"
    MONGO_URI = "mongodb+srv://mefirebase1115_db_user:f76qFi3OqJQsagU2@cluster0.wsppssu.mongodb.net/?appName=Cluster0"
    OWNER_ID = 8149151609
    DB_NAME = "tictactoe_bot"
    GAME_TIMEOUT = 300

settings = Settings()


# ==================== MESSAGE FORMATTER ====================

class MessageFormatter:
    @staticmethod
    def get_rank_emoji(rank: str) -> str:
        rank_emojis = {
            "Beginner": "🌱", "Intermediate": "🌿",
            "Pro": "🌳", "Elite": "👑", "Legend": "⭐"
        }
        return rank_emojis.get(rank, "🎮")

    @staticmethod
    def get_win_message(winner_name: str, loser_name: str, winner_symbol: str) -> str:  
        messages = [  
            f"🎉 **VICTORY ROYALE!** 🎉\n\n🏆 **{winner_name}** ({winner_symbol}) has conquered the board!\n💫 **{loser_name}** fought bravely but fell short.\n\n🎊🎊🎊",  
            f"⚡ **CRUSHING VICTORY!** ⚡\n\n🥇 **{winner_name}** ({winner_symbol}) dominates!\n🥈 **{loser_name}** gave it their best shot.\n\n🎊🎊🎊"  
        ]  
        return random.choice(messages)  
  
    @staticmethod  
    def get_draw_message(player1_name: str, player2_name: str) -> str:  
        return f"🤝 **EPIC STALEMATE!** 🤝\n\n**{player1_name}** vs **{player2_name}**\n\n⚖️ The battle ends in a perfect balance! ⚖️\n\n🎯 No winner, but legends never lose! 🎯"  
  
    @staticmethod  
    def get_game_created_message(player_x_name: str) -> str:  
        return f"🎮 **A NEW CHALLENGE AWAITS!** 🎮\n\n⚔️ **Challenger:** {player_x_name} (❌)\n\n⏳ **Status:** Seeking a worthy opponent...\n\n🎲 Click **Join Game** to accept the challenge!"  
  
    @staticmethod  
    def get_ai_challenge_message(player_name: str, difficulty: str) -> str:  
        difficulty_emojis = {"easy": "😊", "medium": "🤔", "hard": "😈"}  
        return f"🤖 **AI CHALLENGE ACCEPTED!** 🤖\n\n**Human:** {player_name} (❌)\n**AI:** Bot {difficulty_emojis.get(difficulty, '')} ({difficulty.upper()})\n\n⚡ The battle begins!\n🎯 Your move, challenger!"


# ==================== DATABASE LAYER ====================

class Database:
    def __init__(self, mongo_uri: str, db_name: str):
        try:
            logger.info("Connecting to MongoDB...")
            self.client = AsyncIOMotorClient(
                mongo_uri, 
                maxPoolSize=50,
                serverSelectionTimeoutMS=5000,
                retryWrites=True
            )
            # Test connection
            asyncio.get_event_loop().run_until_complete(self.client.server_info())
            self.db = self.client[db_name]
            self.users: AgnosticCollection = self.db["users"]
            self.games: AgnosticCollection = self.db["games"]
            self.groups: AgnosticCollection = self.db["groups"]
            self.bot_stats: AgnosticCollection = self.db["bot_stats"]
            logger.info("✅ Database connected successfully")
        except Exception as e:
            logger.critical(f"❌ Database connection failed: {e}")
            sys.exit(1)

    async def _create_indexes(self):  
        try:  
            logger.info("Creating database indexes...")
            await self.users.create_index("user_id", unique=True)  
            await self.games.create_index("game_id", unique=True)  
            await self.games.create_index("group_id")  
            await self.games.create_index([("group_id", 1), ("state", 1)])  
            await self.games.create_index("created_at")  
            await self.groups.create_index("group_id", unique=True)  
            await self.bot_stats.create_index("_id")  
            logger.info("✅ Database indexes created successfully")  
        except Exception as e:  
            logger.warning(f"⚠️ Index creation warning: {e}")  
  
    async def _init_global_stats(self):  
        await self.bot_stats.update_one(  
            {"_id": "global"},  
            {"$setOnInsert": {  
                "total_users": 0, "total_groups": 0, "total_games": 0,  
                "active_games": 0, "commands_processed": 0, "uptime_hours": 0,  
                "maintenance_mode": False, "blocked_users": []  
            }},  
            upsert=True  
        )  
  
    async def get_user(self, user_id: int) -> Optional[Dict]:  
        return await self.users.find_one({"user_id": user_id})  
  
    async def create_user(self, user: User) -> Dict:  
        user_doc = {  
            "user_id": user.id, "username": user.username,  
            "first_name": user.first_name or "Warrior",  
            "stats": {"wins": 0, "losses": 0, "draws": 0, "games_played": 0, "win_rate": 0.0},  
            "achievements": [], "rank": "Beginner",  
            "created_at": datetime.utcnow(), "last_played": datetime.utcnow()  
        }  
        await self.users.insert_one(user_doc)  
        await self.bot_stats.update_one({"_id": "global"}, {"$inc": {"total_users": 1}})  
        return user_doc  
  
    async def update_user_stats(self, user_id: int, result: str, client: Client = None):  
        user = await self.get_user(user_id)  
        if not user:  
            logger.warning(f"User {user_id} not found for stats update")
            return
      
        old_rank = user["rank"]  
        update_fields = {"stats.games_played": 1}  
      
        if result == "win":  
            update_fields["stats.wins"] = 1  
        elif result == "loss":  
            update_fields["stats.losses"] = 1  
        elif result == "draw":  
            update_fields["stats.draws"] = 1  
      
        await self.users.update_one({"user_id": user_id}, {"$inc": update_fields, "$set": {"last_played": datetime.utcnow()}})  
      
        updated_user = await self.get_user(user_id)  
        wins = updated_user["stats"]["wins"]  
        total = updated_user["stats"]["games_played"]  
        win_rate = (wins / total * 100) if total > 0 else 0  
      
        new_rank = "Beginner"  
        rank_emoji = "🌱"  
        for threshold, rank_name, emoji in [  
            (0, "Beginner", "🌱"), (5, "Intermediate", "🌿"),  
            (20, "Pro", "🌳"), (50, "Elite", "👑"), (100, "Legend", "⭐")  
        ]:  
            if wins >= threshold:  
                new_rank = rank_name  
                rank_emoji = emoji  
      
        await self.users.update_one(  
            {"user_id": user_id},  
            {"$set": {"stats.win_rate": round(win_rate, 2), "rank": new_rank}}  
        )  
      
        if new_rank != old_rank and client:  
            try:  
                await client.send_message(  
                    chat_id=user_id,  
                    text=f"🎊 **RANK UP!** 🎊\n\nYou're now **{rank_emoji} {new_rank.upper()} {rank_emoji}**!"  
                )  
            except Exception as e:  
                logger.warning(f"Failed to send rank up message: {e}")  
  
    async def get_or_create_user(self, user: User) -> Dict:  
        user_data = await self.get_user(user.id)  
        if not user_data:  
            user_data = await self.create_user(user)  
        return user_data  
  
    async def create_game(self, game_id: str, group_id: int, chat_title: str,   
                         player_x: Dict, theme: str = "classic") -> Dict:  
        game_doc = {  
            "game_id": game_id, "group_id": group_id, "chat_title": chat_title,  
            "players": {"X": player_x, "O": None}, "current_turn": "X",  
            "board": self._get_empty_board(), "state": "waiting", "winner": None,  
            "move_count": 0, "theme": theme, "spectators": [],  
            "created_at": datetime.utcnow(), "last_activity": datetime.utcnow()  
        }  
        await self.games.insert_one(game_doc)  
        await self.bot_stats.update_one({"_id": "global"}, {"$inc": {"active_games": 1}})  
        return game_doc  
  
    def _get_empty_board(self) -> List[List[str]]:  
        return [["" for _ in range(3)] for _ in range(3)]  
  
    async def get_active_game(self, group_id: int) -> Optional[Dict]:  
        return await self.games.find_one({  
            "group_id": group_id,  
            "state": {"$in": ["waiting", "playing"]}  
        })  
  
    async def update_game(self, game_id: str, update_data: Dict):  
        update_data["last_activity"] = datetime.utcnow()  
        await self.games.update_one({"game_id": game_id}, {"$set": update_data})  
  
    async def finish_game(self, game_id: str, winner: Optional[str], final_theme: str = None):  
        update_data = {
            "state": "finished", 
            "winner": winner, 
            "last_activity": datetime.utcnow()
        }
        if final_theme:
            update_data["theme"] = final_theme
            
        await self.games.update_one({"game_id": game_id}, {"$set": update_data})  
        await self.bot_stats.update_one(  
            {"_id": "global"},  
            {"$inc": {"active_games": -1, "total_games": 1}}  
        )  
  
    async def get_leaderboard(self, limit: int = 10) -> List[Dict]:  
        cursor = self.users.find().sort([
            ("stats.wins", -1),
            ("stats.games_played", -1),
            ("stats.win_rate", -1)
        ]).limit(limit)
        return await cursor.to_list(length=limit)  
  
    async def get_group_settings(self, group_id: int) -> Dict:  
        group = await self.groups.find_one({"group_id": group_id})  
        if not group:  
            group = {  
                "group_id": group_id, "title": "Unknown",  
                "settings": {"enabled": True, "allow_ai": True, "default_theme": "classic", "auto_pin": False},  
                "stats": {"games_played": 0, "total_players": 0},  
                "created_at": datetime.utcnow()  
            }  
            await self.groups.insert_one(group)  
            await self.bot_stats.update_one({"_id": "global"}, {"$inc": {"total_groups": 1}})  
        return group  
  
    async def update_group_setting(self, group_id: int, setting_path: str, value: Any):  
        await self.groups.update_one(  
            {"group_id": group_id},  
            {"$set": {f"settings.{setting_path}": value}}  
        )  
  
    async def is_user_blocked(self, user_id: int) -> bool:  
        stats = await self.bot_stats.find_one({"_id": "global"})  
        return stats and user_id in stats.get("blocked_users", [])  
  
    async def block_user(self, user_id: int):  
        await self.bot_stats.update_one({"_id": "global"}, {"$addToSet": {"blocked_users": user_id}})  
  
    async def unblock_user(self, user_id: int):  
        await self.bot_stats.update_one({"_id": "global"}, {"$pull": {"blocked_users": user_id}})  
  
    async def get_bot_stats(self) -> Dict:  
        return await self.bot_stats.find_one({"_id": "global"})


# ==================== GAME ENGINE ====================

class GameEngine:
    @staticmethod
    def check_winner(board: List[List[str]]) -> Optional[str]:
        for i in range(3):
            if board[i][0] == board[i][1] == board[i][2] != "":
                return board[i][0]
            if board[0][i] == board[1][i] == board[2][i] != "":
                return board[0][i]
        if board[0][0] == board[1][1] == board[2][2] != "":
            return board[0][0]
        if board[0][2] == board[1][1] == board[2][0] != "":
            return board[0][2]
        return None

    @staticmethod  
    def is_draw(board: List[List[str]]) -> bool:  
        return all(cell != "" for row in board for cell in row)  
  
    @staticmethod  
    def make_move(board: List[List[str]], row: int, col: int, symbol: str) -> bool:  
        if 0 <= row < 3 and 0 <= col < 3 and board[row][col] == "":  
            board[row][col] = symbol  
            return True  
        return False  
  
    @staticmethod  
    def get_available_moves(board: List[List[str]]) -> List[Tuple[int, int]]:  
        return [(i, j) for i in range(3) for j in range(3) if board[i][j] == ""]


# ==================== AI ENGINE ====================

class AIEngine:
    @staticmethod
    async def get_move(board: List[List[str]], difficulty: str, ai_symbol: str, player_symbol: str) -> Tuple[int, int]:
        try:
            if difficulty == "easy":
                return await AIEngine._random_move(board)
            elif difficulty == "medium":
                return await AIEngine._blocking_move(board, ai_symbol, player_symbol)
            elif difficulty == "hard":
                return await AIEngine._minimax_move(board, ai_symbol, player_symbol)
            else:
                return await AIEngine._random_move(board)
        except Exception as e:
            logger.error(f"AI move error: {e}", exc_info=True)
            import random
            moves = GameEngine.get_available_moves(board)
            return random.choice(moves) if moves else (0, 0)

    @staticmethod  
    async def _random_move(board: List[List[str]]) -> Tuple[int, int]:  
        import random  
        moves = GameEngine.get_available_moves(board)  
        return random.choice(moves) if moves else (0, 0)  
  
    @staticmethod  
    async def _blocking_move(board: List[List[str]], ai_symbol: str, player_symbol: str) -> Tuple[int, int]:  
        for row, col in GameEngine.get_available_moves(board):  
            test_board = [r[:] for r in board]  
            test_board[row][col] = ai_symbol  
            if GameEngine.check_winner(test_board) == ai_symbol:  
                return row, col  
      
        for row, col in GameEngine.get_available_moves(board):  
            test_board = [r[:] for r in board]  
            test_board[row][col] = player_symbol  
            if GameEngine.check_winner(test_board) == player_symbol:  
                return row, col  
      
        import random  
        return random.choice(GameEngine.get_available_moves(board))  
  
    @staticmethod  
    async def _minimax_move(board: List[List[str]], ai_symbol: str, player_symbol: str) -> Tuple[int, int]:  
        best_score = float('-inf')  
        best_move = None  
          
        for row, col in GameEngine.get_available_moves(board):  
            test_board = [r[:] for r in board]  
            test_board[row][col] = ai_symbol  
              
            score = AIEngine._minimax(  
                test_board, 0, False,   
                ai_symbol, player_symbol,  
                float('-inf'), float('inf')  
            )  
              
            if score > best_score:  
                best_score = score  
                best_move = (row, col)  
      
        return best_move or await AIEngine._random_move(board)  
  
    @staticmethod  
    def _minimax(board: List[List[str]], depth: int, is_maximizing: bool,  
                 ai_symbol: str, player_symbol: str, alpha: float, beta: float) -> int:  
        winner = GameEngine.check_winner(board)  
        if winner == ai_symbol:  
            return 10 - depth  
        elif winner == player_symbol:  
            return depth - 10  
        elif GameEngine.is_draw(board):  
            return 0  
      
        if is_maximizing:  
            max_eval = float('-inf')  
            for row, col in GameEngine.get_available_moves(board):  
                board[row][col] = ai_symbol  
                eval = AIEngine._minimax(board, depth + 1, False,   
                                       ai_symbol, player_symbol, alpha, beta)  
                board[row][col] = ""  
                max_eval = max(max_eval, eval)  
                alpha = max(alpha, eval)  
                if beta <= alpha:  
                    break  
            return max_eval  
        else:  
            min_eval = float('inf')  
            for row, col in GameEngine.get_available_moves(board):  
                board[row][col] = player_symbol  
                eval = AIEngine._minimax(board, depth + 1, True,   
                                       ai_symbol, player_symbol, alpha, beta)  
                board[row][col] = ""  
                min_eval = min(min_eval, eval)  
                beta = min(beta, eval)  
                if beta <= alpha:  
                    break  
            return min_eval


# ==================== UI KEYBOARDS ====================

class UIKeyboard:
    @staticmethod
    def get_main_menu():
        return InlineKeyboardMarkup([
            [
                InlineKeyboardButton("➕ Add to Group", url=f"https://t.me/TicTacToeBot?startgroup=true"),
                InlineKeyboardButton("📖 Help", callback_data="help")
            ]
        ])

    @staticmethod  
    def get_game_board(board: List[List[str]], theme: str,   
                      game_state: str, current_turn: str,   
                      players: Dict, message_id: Optional[int] = None) -> InlineKeyboardMarkup:  
        theme_data = {  
            "classic": {"X": "❌", "O": "⭕", "empty": "⬜"},  
            "colorful": {"X": "🔴", "O": "🔵", "empty": "⚪"},  
            "nature": {"X": "🌳", "O": "💧", "empty": "🌱"},  
            "space": {"X": "🚀", "O": "🛸", "empty": "⭐"},  
            "food": {"X": "🍕", "O": "🍔", "empty": "🧀"},  
            "royal": {"X": "👑", "O": "💎", "empty": "🟦"},  
            "warrior": {"X": "⚔️", "O": "🛡️", "empty": "🏟️"},  
            "magical": {"X": "🔮", "O": "✨", "empty": "🌟"},  
            "pirate": {"X": "☠️", "O": "🏴‍☠️", "empty": "🌊"},  
            "ninja": {"X": "🥷", "O": "🗡️", "empty": "🌫️"}  
        }.get(theme, {"X": "❌", "O": "⭕", "empty": "⬜"})  
          
        buttons = []  
        for i in range(3):  
            row = []  
            for j in range(3):  
                cell = board[i][j]  
                emoji = theme_data.get(cell, theme_data["empty"])  
                row.append(InlineKeyboardButton(  
                    emoji,  
                    callback_data=f"move_{i}_{j}" if game_state == "playing" else "noop"  
                ))  
            buttons.append(row)  
      
        # Status row  
        if game_state == "waiting":  
            status_text = "⏳ Awaiting challenger..."  
        elif game_state == "playing":  
            current_player = players[current_turn]  
            name = current_player["first_name"] if current_player else "Unknown"  
            status_text = f"🎮 {name}'s turn ({current_turn})"  
        else:  
            status_text = "🏁 Battle Complete!"  
      
        buttons.append([InlineKeyboardButton(status_text, callback_data="noop")])  
          
        # Action buttons  
        action_row = []  
        if game_state == "waiting":  
            action_row.append(InlineKeyboardButton("🎮 Join Game", callback_data="join_game"))  
        elif game_state == "finished":  
            action_row.append(InlineKeyboardButton("🔄 New Game", callback_data="newgame"))  
      
        action_row.append(InlineKeyboardButton("🎨 Themes", callback_data="theme_menu"))  
          
        if action_row:  
            buttons.append(action_row)  
      
        return InlineKeyboardMarkup(buttons)  
  
    @staticmethod  
    def get_theme_menu(current_theme: str) -> InlineKeyboardMarkup:  
        themes = ["classic", "colorful", "nature", "space", "food",   
                 "royal", "warrior", "magical", "pirate", "ninja"]  
      
        buttons = []  
        row = []  
      
        for idx, theme in enumerate(themes):  
            label = f"✅ {theme.title()}" if theme == current_theme else theme.title()  
            row.append(InlineKeyboardButton(label, callback_data=f"set_theme_{theme}"))  
          
            if len(row) == 2:  
                buttons.append(row)  
                row = []  
      
        if row:  
            buttons.append(row)  
      
        buttons.append([InlineKeyboardButton("🔙 Back", callback_data="back_to_game")])  
        return InlineKeyboardMarkup(buttons)  
  
    @staticmethod  
    def get_admin_menu(group_settings: Dict) -> InlineKeyboardMarkup:  
        enabled_emoji = "✅" if group_settings.get("enabled", True) else "❌"  
        ai_emoji = "✅" if group_settings.get("allow_ai", True) else "❌"  
        pin_emoji = "✅" if group_settings.get("auto_pin", False) else "❌"  
      
        return InlineKeyboardMarkup([  
            [  
                InlineKeyboardButton(f"{enabled_emoji} Bot Enabled", callback_data="toggle_enabled"),  
                InlineKeyboardButton(f"{ai_emoji} AI Allowed", callback_data="toggle_ai")  
            ],  
            [  
                InlineKeyboardButton(f"{pin_emoji} Auto Pin", callback_data="toggle_pin"),  
                InlineKeyboardButton("🎨 Default Theme", callback_data="default_theme_menu")  
            ],  
            [InlineKeyboardButton("🗑 Reset Game", callback_data="reset_game")],  
            [InlineKeyboardButton("🔙 Close", callback_data="close_menu")]  
        ])


# ==================== ANTI-SPAM ====================

class AntiSpam:
    def __init__(self):
        self.callback_cache = {}
        self.user_last_action = {}
        self.last_cleanup = time.time()

    def _cleanup_old_entries(self):  
        now = time.time()  
        for user_id in list(self.callback_cache.keys()):  
            self.callback_cache[user_id] = [ts for ts in self.callback_cache[user_id] if now - ts < 60]  
            if not self.callback_cache[user_id]:  
                del self.callback_cache[user_id]  
      
        self.user_last_action = {uid: ts for uid, ts in self.user_last_action.items() if now - ts < 60}  
        self.last_cleanup = now  
  
    def is_allowed(self, user_id: int) -> bool:  
        now = time.time()  
        if now - self.last_cleanup > 3600:  
            self._cleanup_old_entries()  
      
        self.callback_cache[user_id] = [ts for ts in self.callback_cache.get(user_id, []) if now - ts < 60]  
      
        if len(self.callback_cache.get(user_id, [])) >= 30:  
            return False  
      
        if user_id in self.user_last_action and now - self.user_last_action[user_id] < 1:  
            return False  
      
        self.callback_cache.setdefault(user_id, []).append(now)  
        self.user_last_action[user_id] = now  
        return True


# ==================== BOT CORE ====================

class BotCore:
    def __init__(self, client: Client, db: Database):
        self.client = client
        self.db = db
        self.game_engine = GameEngine()
        self.ai_engine = AIEngine()
        self.ui = UIKeyboard()
        self.anti_spam = AntiSpam()
        self.cleanup_task = None
        self.game_locks = {}
        self.ai_tasks = {}

    async def start_cleanup_task(self):  
        async def cleanup():  
            while True:  
                try:  
                    await self._cleanup_expired_games()  
                except Exception as e:  
                    logger.error(f"Cleanup error: {e}", exc_info=True)  
                await asyncio.sleep(60)  
      
        self.cleanup_task = asyncio.create_task(cleanup())  
        logger.info("✅ Cleanup task started")
  
    async def _cleanup_expired_games(self):  
        expired_time = datetime.utcnow() - timedelta(seconds=settings.GAME_TIMEOUT)  
        async for game in self.db.games.find({  
            "state": "waiting",  
            "created_at": {"$lt": expired_time}  
        }):  
            await self.db.finish_game(game["game_id"], None, game.get("theme"))  
            try:  
                await self.client.edit_message_text(  
                    chat_id=game["group_id"],  
                    message_id=game.get("message_id"),  
                    text="⏰ **Game expired!** No challenger appeared in time.",  
                    reply_markup=None  
                )  
            except Exception as e:  
                logger.warning(f"Failed to edit expired game message: {e}")  
            
            # Clean up locks
            self.game_locks.pop(game["group_id"], None)
            self.ai_tasks.pop(game["game_id"], None)

    def is_owner(self, user_id: int) -> bool:  
        return user_id == settings.OWNER_ID  
  
    async def is_group_admin(self, chat, user_id: int) -> bool:  
        try:  
            member = await chat.get_member(user_id)  
            return member.status in [enums.ChatMemberStatus.OWNER, enums.ChatMemberStatus.ADMINISTRATOR]  
        except Exception as e:  
            logger.warning(f"Admin check failed: {e}")  
            return False  
  
    async def is_maintenance_mode(self) -> bool:  
        stats = await self.db.get_bot_stats()  
        return stats.get("maintenance_mode", False)  
  
    # ==================== COMMAND HANDLERS ====================  
    async def handle_start(self, client: Client, message: Message):  
        logger.info(f"[COMMAND] /start from user {message.from_user.id} in {message.chat.type}")
        if message.chat.type == enums.ChatType.PRIVATE:  
            await message.reply(  
                "**🎮 Welcome to Tic-Tac-Toe Pro!**\n\n"  
                "Play classic Tic-Tac-Toe with friends or challenge our AI.\n\n"  
                "**Features:**\n"  
                "✅ Multiplayer & AI modes\n"  
                "🏆 Leaderboard & ranks\n"  
                "🎨 Multiple themes\n"  
                "📊 Detailed stats",  
                reply_markup=self.ui.get_main_menu()  
            )  
        else:  
            await message.reply("✅ Bot is active! Use /newgame to start.")  
  
    async def handle_newgame(self, client: Client, message: Message):  
        logger.info(f"[COMMAND] /newgame from user {message.from_user.id} in group {message.chat.id}")
        if message.chat.type not in [enums.ChatType.GROUP, enums.ChatType.SUPERGROUP]:  
            await message.reply("❌ This command only works in groups!")  
            return  
      
        group_id = message.chat.id  
        user = message.from_user  
      
        if await self.is_maintenance_mode():  
            await message.reply("🛠 Bot is under maintenance. Please try again later.")  
            return  
      
        if await self.db.is_user_blocked(user.id):  
            await message.reply("❌ You are blocked from using this bot.")  
            return  
      
        group_settings = await self.db.get_group_settings(group_id)  
        if not group_settings["settings"]["enabled"]:  
            await message.reply("❌ Bot is disabled in this group.")  
            return  
      
        if await self.db.get_active_game(group_id):  
            await message.reply("❌ A game is already active in this group!")  
            return  
      
        game_id = f"g_{group_id}_{int(time.time())}"  
        player_x = {  
            "user_id": user.id,  
            "username": user.username,  
            "first_name": user.first_name or "Warrior"  
        }  
      
        game = await self.db.create_game(game_id, group_id, message.chat.title, player_x)  
      
        msg = await message.reply(  
            MessageFormatter.get_game_created_message(player_x['first_name']),  
            reply_markup=InlineKeyboardMarkup([  
                [InlineKeyboardButton("🎮 Join Game", callback_data="join_game")],  
                [InlineKeyboardButton("📨 Invite Friend",   
                    url=f"https://t.me/{client.me.username}?start={game_id}")],  
                [InlineKeyboardButton("🎨 Themes", callback_data="theme_menu")],  
                [InlineKeyboardButton("❓ How to Play", callback_data="help")]  
            ])  
        )  
      
        await self.db.update_game(game_id, {"message_id": msg.id})  
  
    async def handle_ai(self, client: Client, message: Message):  
        logger.info(f"[COMMAND] /ai from user {message.from_user.id} in group {message.chat.id}")
        if message.chat.type not in [enums.ChatType.GROUP, enums.ChatType.SUPERGROUP]:  
            await message.reply("❌ This command only works in groups!")  
            return  
      
        group_id = message.chat.id  
        user = message.from_user  
      
        if await self.is_maintenance_mode():  
            await message.reply("🛠 Bot is under maintenance. Please try again later.")  
            return  
      
        if await self.db.is_user_blocked(user.id):  
            await message.reply("❌ You are blocked from using this bot.")  
            return  
      
        group_settings = await self.db.get_group_settings(group_id)  
        if not group_settings["settings"]["enabled"]:  
            await message.reply("❌ Bot is disabled in this group.")  
            return  
        if not group_settings["settings"]["allow_ai"]:  
            await message.reply("❌ AI games are disabled in this group.")  
            return  
      
        if await self.db.get_active_game(group_id):  
            await message.reply("❌ A game is already active!")  
            return  
      
        args = message.text.split()  
        difficulty = "medium"  
        if len(args) > 1:  
            diff_arg = args[1].lower()  
            if diff_arg in ["easy", "medium", "hard"]:  
                difficulty = diff_arg  
      
        game_id = f"ai_{group_id}_{int(time.time())}"  
        player_x = {  
            "user_id": user.id,  
            "username": user.username,  
            "first_name": user.first_name or "Warrior"  
        }  
        player_o = {  
            "user_id": client.me.id,  
            "username": client.me.username,  
            "first_name": "AI Bot"  
        }  
      
        game = await self.db.create_game(game_id, group_id, message.chat.title, player_x)  
        game["players"]["O"] = player_o  
        game["state"] = "playing"  
        game["ai_difficulty"] = difficulty  
        game["ai_symbol"] = "O"  
        game["player_symbol"] = "X"  
      
        await self.db.update_game(game_id, {  
            "players": game["players"],  
            "state": "playing"  
        })  
      
        msg = await message.reply(  
            MessageFormatter.get_ai_challenge_message(player_x['first_name'], difficulty),  
            reply_markup=self.ui.get_game_board(  
                game["board"], game["theme"],   
                game["state"], game["current_turn"],   
                game["players"]  
            )  
        )  
      
        await self.db.update_game(game_id, {"message_id": msg.id})  
  
    async def handle_profile(self, client: Client, message: Message):  
        logger.info(f"[COMMAND] /profile from user {message.from_user.id}")
        user = message.from_user  
        user_data = await self.db.get_or_create_user(user)  
      
        stats = user_data["stats"]  
        rank = user_data["rank"]   
        rank_emoji = MessageFormatter.get_rank_emoji(rank)  
      
        # Calculate streak  
        streak = min(stats["wins"] // 5, 10)  
        streak_fire = "🔥" * streak  
      
        # Calculate win rate bar  
        win_rate = int(stats['win_rate'])  
        filled_bars = win_rate // 10  
        win_rate_bar = "🟩" * filled_bars + "⬜" * (10 - filled_bars)  
      
        await message.reply(  
            f"👤 **{user_data['first_name']}'s Battle Profile**\n\n"  
            f"**Rank:** {rank_emoji} **{rank.upper()}** {rank_emoji}\n"  
            f"**Streak:** {streak_fire or '❄️ None'}\n\n"  
            f"**⚔️ Battle Statistics:**\n"  
            f"🏆 **Victories:** `{stats['wins']}`\n"  
            f"💔 **Defeats:** `{stats['losses']}`\n"   
            f"🤝 **Stalemates:** `{stats['draws']}`\n"  
            f"📊 **Total Battles:** `{stats['games_played']}`\n"  
            f"💯 **Win Rate:** `{win_rate}%`\n"  
            f"{win_rate_bar}\n\n"  
            f"🎮 Keep playing to reach the next rank!"  
        )  
  
    async def handle_leaderboard(self, client: Client, message: Message):  
        logger.info(f"[COMMAND] /leaderboard from user {message.from_user.id}")
        leaderboard = await self.db.get_leaderboard(10)  
      
        if not leaderboard:  
            await message.reply("❌ The arena is empty! Be the first warrior!")  
            return  
      
        text = "🏆 **GLOBAL HALL OF FAME - TOP 10** 🏆\n\n"  
      
        for idx, player in enumerate(leaderboard, 1):  
            name = player['first_name'][:15]  
            wins = player['stats']['wins']  
            rank = player['rank']  
            rank_emoji = MessageFormatter.get_rank_emoji(rank)  
          
            if idx == 1:  
                medal = "👑🥇"  
            elif idx == 2:  
                medal = "🥈"   
            elif idx == 3:  
                medal = "🥉"  
            else:  
                medal = f"#{idx}"  
          
            text += f"{medal} **{name}** - `{wins}` wins {rank_emoji}\n"  
      
        text += f"\n⚔️ Climb the ranks to claim your spot!"  
        await message.reply(text)  
  
    async def handle_help(self, client: Client, message: Message):  
        logger.info(f"[COMMAND] /help from user {message.from_user.id}")
        help_text = (  
            "🎮 **TIC-TAC-TOE PRO - COMMAND GUIDE** 🎮\n\n"  
          
            "⚔️ **BATTLE COMMANDS**\n"  
            "▸ `/newgame` - Start multiplayer duel\n"  
            "▸ `/ai [easy/medium/hard]` - Challenge the AI\n"  
            "▸ `/profile` - View your warrior stats\n"  
            "▸ `/leaderboard` - See top champions\n"  
            "\n"  
            "🛡️ **ADMIN COMMANDS**\n"  
            "▸ `/settings` - Configure group settings\n"  
            "▸ `/resetgame` - Force stop current game\n"  
            "▸ `/enablebot` - Activate bot in group\n"  
            "▸ `/disablebot` - Deactivate bot\n"  
            "\n"  
            "🎯 **HOW TO PLAY**\n"  
            "1. Tap board cells to place your mark\n"  
            "2. Get 3 in a row to win!\n"  
            "3. Block your opponent's moves\n"  
            "4. Use themes for visual flair\n"  
            "\n"  
            "✨ Change themes during gameplay!\n"  
            "📈 Track your stats and rank up!"  
        )  
        await message.reply(help_text)  
  
    async def handle_resetgame(self, client: Client, message: Message):  
        logger.info(f"[COMMAND] /resetgame from user {message.from_user.id} in group {message.chat.id}")
        if message.chat.type not in [enums.ChatType.GROUP, enums.ChatType.SUPERGROUP]:  
            return  
      
        chat = message.chat  
        user = message.from_user  
      
        if not await self.is_group_admin(chat, user.id) and not self.is_owner(user.id):  
            await message.reply("❌ Only admins can use this command!")  
            return  
      
        group_id = chat.id  
        game = await self.db.get_active_game(group_id)  
        if not game:  
            await message.reply("❌ No active game to reset!")  
            return  
      
        await self.db.finish_game(game["game_id"], None, game.get("theme"))  
      
        try:  
            await self.client.edit_message_text(  
                chat_id=group_id,  
                message_id=game.get("message_id"),  
                text="🗑 **Game has been reset by admin.**\n\nUse /newgame to start a new battle!",  
                reply_markup=None  
            )  
        except Exception as e:  
            logger.warning(f"Failed to edit reset message: {e}")  
      
        # Clean up locks
        self.game_locks.pop(group_id, None)
        self.ai_tasks.pop(game["game_id"], None)
      
        await message.reply("✅ Game reset successfully!")  
  
    async def handle_settings(self, client: Client, message: Message):  
        logger.info(f"[COMMAND] /settings from user {message.from_user.id} in group {message.chat.id}")
        if message.chat.type not in [enums.ChatType.GROUP, enums.ChatType.SUPERGROUP]:  
            return  
      
        chat = message.chat  
        user = message.from_user  
      
        if not await self.is_group_admin(chat, user.id) and not self.is_owner(user.id):  
            await message.reply("❌ Only admins can use this command!")  
            return  
      
        group_settings = await self.db.get_group_settings(chat.id)  
        await message.reply(  
            "**⚙️ Group Settings**\n\nConfigure bot behavior for this group:",  
            reply_markup=self.ui.get_admin_menu(group_settings["settings"])  
        )  
  
    async def handle_enablebot(self, client: Client, message: Message):  
        logger.info(f"[COMMAND] /enablebot from user {message.from_user.id} in group {message.chat.id}")
        if message.chat.type not in [enums.ChatType.GROUP, enums.ChatType.SUPERGROUP]:  
            return  
      
        chat = message.chat  
        user = message.from_user  
      
        if not await self.is_group_admin(chat, user.id) and not self.is_owner(user.id):  
            await message.reply("❌ Only admins can use this command!")  
            return  
      
        await self.db.update_group_setting(chat.id, "enabled", True)  
        await message.reply("✅ Bot enabled in this group!")  
  
    async def handle_disablebot(self, client: Client, message: Message):  
        logger.info(f"[COMMAND] /disablebot from user {message.from_user.id} in group {message.chat.id}")
        if message.chat.type not in [enums.ChatType.GROUP, enums.ChatType.SUPERGROUP]:  
            return  
      
        chat = message.chat  
        user = message.from_user  
      
        if not await self.is_group_admin(chat, user.id) and not self.is_owner(user.id):  
            await message.reply("❌ Only admins can use this command!")  
            return  
      
        await self.db.update_group_setting(chat.id, "enabled", False)  
        await message.reply("❌ Bot disabled in this group!")  
  
    async def handle_owner_stats(self, client: Client, message: Message):  
        if not self.is_owner(message.from_user.id):  
            return  
      
        stats = await self.db.get_bot_stats()  
      
        uptime = stats.get("uptime_hours", 0)  
        maintenance = "🛠 ON" if stats.get("maintenance_mode", False) else "✅ OFF"  
      
        await message.reply(  
            f"**👑 Bot Owner Statistics**\n\n"  
            f"**System:**\n"  
            f"⏱ Uptime: {uptime:.1f} hours\n"  
            f"🛠 Maintenance Mode: {maintenance}\n\n"  
            f"**Usage:**\n"  
            f"👥 Total Users: {stats['total_users']}\n"  
            f"👥 Total Groups: {stats['total_groups']}\n"  
            f"🎮 Total Games: {stats['total_games']}\n"  
            f"🎮 Active Games: {stats['active_games']}\n"  
            f"📥 Commands Processed: {stats['commands_processed']}\n\n"  
            f"**Security:**\n"  
            f"🚫 Blocked Users: {len(stats.get('blocked_users', []))}"  
        )  
  
    async def handle_broadcast(self, client: Client, message: Message):  
        if not self.is_owner(message.from_user.id):  
            return  
      
        if len(message.text) <= 10:  
            await message.reply("❌ Usage: /broadcast <message>")  
            return  
      
        text = message.text[11:].strip()  
        if not text:  
            await message.reply("❌ Provide a message to broadcast!")  
            return  
      
        cursor = self.db.groups.find({})  
        success = 0  
        failed = 0  
      
        status_msg = await message.reply(f"📢 Broadcasting to groups... 0/0")  
      
        async for group in cursor:  
            try:  
                await self.client.send_message(  
                    chat_id=group["group_id"],  
                    text=f"📢 **Broadcast Message**\n\n{text}",  
                    disable_notification=True  
                )  
                success += 1  
            except Exception as e:  
                logger.warning(f"Broadcast failed for {group['group_id']}: {e}")  
                failed += 1  
          
            if (success + failed) % 10 == 0:  
                try:  
                    await status_msg.edit(f"📢 Broadcasting... {success}/{success + failed}")  
                except:  
                    pass  
          
            await asyncio.sleep(0.1)  
      
        await status_msg.edit(f"✅ Broadcast complete!\nSuccess: {success}\nFailed: {failed}")  
  
    async def handle_active_games(self, client: Client, message: Message):  
        if not self.is_owner(message.from_user.id):  
            return  
      
        games = self.db.games.find({"state": {"$in": ["waiting", "playing"]}}).limit(50)  
      
        if not games:  
            await message.reply("❌ No active games found!")  
            return  
      
        text = "**🎮 Active Games**\n\n"  
        count = 0  
      
        async for game in games:  
            if count >= 20:  
                break  
          
            group_name = game.get("chat_title", "Unknown")  
            state = game["state"]  
            players = game["players"]  
          
            player_x = players["X"]["first_name"] if players["X"] else "None"  
            player_o = players["O"]["first_name"] if players["O"] else "Waiting"  
          
            text += f"**{group_name}**\n"  
            text += f"State: {state} | X: {player_x} | O: {player_o}\n\n"  
            count += 1  
      
        await message.reply(text)  
  
    async def handle_users(self, client: Client, message: Message):  
        if not self.is_owner(message.from_user.id):  
            return  
      
        count = await self.db.users.count_documents({})  
        await message.reply(f"👥 **Total Users:** {count}")  
  
    async def handle_groups(self, client: Client, message: Message):  
        if not self.is_owner(message.from_user.id):  
            return  
      
        count = await self.db.groups.count_documents({})  
        await message.reply(f"👥 **Total Groups:** {count}")  
  
    async def handle_restart(self, client: Client, message: Message):  
        if not self.is_owner(message.from_user.id):  
            return  
      
        await message.reply("🔄 Restarting bot...")  
        await client.stop()  
        sys.exit(0)  
  
    async def handle_shutdown(self, client: Client, message: Message):  
        if not self.is_owner(message.from_user.id):  
            return  
      
        await message.reply("⛔ Shutting down bot...")  
        await client.stop()  
        sys.exit(0)  
  
    async def handle_maintenance(self, client: Client, message: Message):  
        if not self.is_owner(message.from_user.id):  
            return  
      
        args = message.text.split()  
        if len(args) < 2:  
            await message.reply("❌ Usage: /maintenance on/off")  
            return  
      
        status = args[1].lower()  
        if status not in ["on", "off"]:  
            await message.reply("❌ Usage: /maintenance on/off")  
            return  
      
        mode = status == "on"  
        await self.db.bot_stats.update_one(  
            {"_id": "global"},  
            {"$set": {"maintenance_mode": mode}}  
        )  
      
        state = "🛠 ON" if mode else "✅ OFF"  
        await message.reply(f"🛠 Maintenance mode: **{state}**")  
  
    async def handle_block(self, client: Client, message: Message):  
        if not self.is_owner(message.from_user.id):  
            return  
      
        args = message.text.split()  
        if len(args) < 2:  
            await message.reply("❌ Usage: /block <user_id>")  
            return  
      
        try:  
            user_id = int(args[1])  
            await self.db.block_user(user_id)  
            await message.reply(f"🚫 User {user_id} has been blocked.")  
        except ValueError:  
            await message.reply("❌ Invalid user ID!")  
  
    async def handle_unblock(self, client: Client, message: Message):  
        if not self.is_owner(message.from_user.id):  
            return  
      
        args = message.text.split()  
        if len(args) < 2:  
            await message.reply("❌ Usage: /unblock <user_id>")  
            return  
      
        try:  
            user_id = int(args[1])  
            await self.db.unblock_user(user_id)  
            await message.reply(f"✅ User {user_id} has been unblocked.")  
        except ValueError:  
            await message.reply("❌ Invalid user ID!")  
  
    # ==================== CALLBACK HANDLERS ====================  
    async def handle_callback(self, client: Client, query: CallbackQuery):  
        user = query.from_user  
        logger.info(f"[CALLBACK] {query.data} from user {user.id}")
      
        # Anti-spam check  
        if not self.anti_spam.is_allowed(user.id):  
            await query.answer("⏱ Too fast! Slow down.", show_alert=True)  
            return  
      
        # Update command stats  
        await self.db.bot_stats.update_one(  
            {"_id": "global"},  
            {"$inc": {"commands_processed": 1}}  
        )  
      
        data = query.data  
      
        # Route callbacks  
        try:
            if data.startswith("move_"):  
                lock = self.game_locks.setdefault(query.message.chat.id, asyncio.Lock())  
                async with lock:  
                    await self._handle_move(client, query)  
            elif data == "join_game":  
                await self._handle_join(client, query)  
            elif data == "rematch":  
                await self._handle_rematch(client, query)  
            elif data == "theme_menu":  
                await self._handle_theme_menu(client, query)  
            elif data.startswith("set_theme_"):  
                await self._handle_set_theme(client, query)  
            elif data == "back_to_game":  
                await self._handle_back_to_game(client, query)  
            elif data == "help":  
                await self._handle_help_callback(client, query)  
            elif data == "newgame":  
                await self._handle_newgame_callback(client, query)  
            elif data == "noop":  
                await query.answer()  
            # Admin callbacks  
            elif data == "toggle_enabled":  
                await self._handle_toggle_enabled(client, query)  
            elif data == "toggle_ai":  
                await self._handle_toggle_ai(client, query)  
            elif data == "toggle_pin":  
                await self._handle_toggle_pin(client, query)  
            elif data == "default_theme_menu":  
                await self._handle_default_theme_menu(client, query)  
            elif data == "reset_game":  
                await self._handle_reset_game_callback(client, query)  
            elif data == "close_menu":  
                await self._handle_close_menu(client, query)
            else:
                logger.warning(f"Unknown callback data: {data}")
                await query.answer("❌ Unknown action", show_alert=True)
        except Exception as e:
            logger.error(f"Callback error: {e}", exc_info=True)
            await query.answer("❌ An error occurred. Please try again.", show_alert=True)  
  
    async def _handle_move(self, client: Client, query: CallbackQuery):  
        user = query.from_user  
        chat = query.message.chat  
      
        game = await self.db.get_active_game(chat.id)  
        if not game:  
            await query.answer("❌ No active game!", show_alert=True)  
            return  
      
        if game["state"] != "playing":  
            await query.answer("❌ Game is not active!", show_alert=True)  
            return  
      
        current_player = game["players"][game["current_turn"]]  
        if not current_player or current_player["user_id"] != user.id:  
            await query.answer("⏳ Not your turn!", show_alert=True)  
            return  
      
        try:
            _, row, col = query.data.split("_")  
            row, col = int(row), int(col)
            if not (0 <= row < 3 and 0 <= col < 3):
                raise ValueError
        except:
            await query.answer("❌ Invalid move!", show_alert=True)
            return
      
        board = game["board"]  
        if not self.game_engine.make_move(board, row, col, game["current_turn"]):  
            await query.answer("❌ Invalid move!", show_alert=True)  
            return  
      
        game["move_count"] += 1  
      
        winner = self.game_engine.check_winner(board)  
        if winner:  
            game["state"] = "finished"  
            await self.db.finish_game(game["game_id"], winner, game.get("theme"))  
          
            players = game["players"]  
            winner_player = players[winner]  
            loser_symbol = "O" if winner == "X" else "X"  
            loser_player = players.get(loser_symbol)  
          
            if winner_player and loser_player:  
                # Notify winner  
                if winner_player["user_id"] != client.me.id:  
                    await self.db.update_user_stats(winner_player["user_id"], "win", client)  
              
                # Notify loser  
                if loser_player["user_id"] != client.me.id:  
                    await self.db.update_user_stats(loser_player["user_id"], "loss", client)  
                  
                    try:  
                        loser_user = await self.db.get_user(loser_player["user_id"])  
                        await client.send_message(  
                            chat_id=loser_player["user_id"],  
                            text=(  
                                f"💔 **Tough battle!** 💔\n\n"  
                                f"You lost to **{winner_player['first_name']}**, but you're learning!\n"  
                                f"🎮 Keep practicing and you'll be victorious next time!\n\n"  
                                f"⭐ Your current rank: {loser_user['rank']}"  
                            )  
                        )  
                    except:  
                        pass  
              
                await query.message.reply(  
                    MessageFormatter.get_win_message(  
                        winner_player["first_name"],  
                        loser_player["first_name"],  
                        winner  
                    )  
                )  
          
        elif self.game_engine.is_draw(board):  
            game["state"] = "finished"  
            await self.db.finish_game(game["game_id"], None, game.get("theme"))  
          
            p1_name = game["players"]["X"]["first_name"] if game["players"]["X"] else "Player X"  
            p2_name = game["players"]["O"]["first_name"] if game["players"]["O"] else "Player O"  
          
            await query.message.reply(MessageFormatter.get_draw_message(p1_name, p2_name))  
          
            for player in game["players"].values():  
                if player and player["user_id"] != client.me.id:  
                    await self.db.update_user_stats(player["user_id"], "draw", client)  
      
        else:  
            game["current_turn"] = "O" if game["current_turn"] == "X" else "X"  
      
        await self.db.update_game(game["game_id"], {  
            "board": board,  
            "current_turn": game["current_turn"],  
            "state": game["state"],  
            "move_count": game["move_count"]  
        })  
      
        try:  
            await query.message.edit_reply_markup(  
                reply_markup=self.ui.get_game_board(  
                    board, game["theme"], game["state"],   
                    game["current_turn"], game["players"]  
                )  
            )  
        except MessageNotModified:  
            pass  
      
        await query.answer()  
      
        # AI move if needed  
        if game["state"] == "playing" and game["current_turn"] == game.get("ai_symbol"):  
            task = asyncio.create_task(self._handle_ai_move(client, query.message, game))
            self.ai_tasks[game["game_id"]] = task  
  
    async def _handle_ai_move(self, client: Client, message: Message, game: Dict):  
        # Notify user that AI is thinking  
        try:  
            await message.reply_chat_action(enums.ChatAction.TYPING)  
        except:  
            pass  
          
        await asyncio.sleep(1.5)  
      
        ai_symbol = game["ai_symbol"]  
        player_symbol = game["player_symbol"]  
        difficulty = game["ai_difficulty"]  
      
        try:
            row, col = await self.ai_engine.get_move(  
                game["board"], difficulty, ai_symbol, player_symbol  
            )  
      
            if self.game_engine.make_move(game["board"], row, col, ai_symbol):  
                game["move_count"] += 1  
          
                winner = self.game_engine.check_winner(game["board"])  
                if winner:  
                    game["state"] = "finished"  
                    await self.db.finish_game(game["game_id"], winner, game.get("theme"))  
              
                    if winner == game["ai_symbol"]:  
                        await message.reply(  
                            f"🤖 **THE MACHINE PREVAILS!** 🤖\n\n"  
                            f"😈 The AI ({game['ai_difficulty']}) has bested you!\n"  
                            f"💡 **Tip:** Try a lower difficulty or practice more!\n\n"  
                            f"🔄 Challenge again when ready!"  
                        )  
                    else:  
                        await message.reply(  
                            f"🎉 **HUMAN TRIUMPH!** 🎉\n\n"  
                            f"🏆 **Congratulations!** You defeated the {game['ai_difficulty']} AI!\n"  
                            f"🌟 Truly impressive!\n\n"  
                            f"😎 You're becoming a Tic-Tac-Toe master!"  
                        )  
                  
                    player = game["players"]["X"]  
                    if player and player["user_id"] != self.client.me.id:  
                        result = "win" if winner != game["ai_symbol"] else "loss"  
                        await self.db.update_user_stats(player["user_id"], result, self.client)  
              
                elif self.game_engine.is_draw(game["board"]):  
                    game["state"] = "finished"  
                    await self.db.finish_game(game["game_id"], None, game.get("theme"))  
              
                    await message.reply(  
                        f"🤝 **NO VICTORY, BUT HONOR!** 🤝\n\n"  
                        f"You fought the {game['ai_difficulty']} AI to a draw!\n"  
                        f"🎭 Respectable performance!\n\n"  
                        f"🔥 Try again to secure the win!"  
                    )  
              
                    for player in game["players"].values():  
                        if player and player["user_id"] != self.client.me.id:  
                            await self.db.update_user_stats(player["user_id"], "draw", self.client)  
              
                else:  
                    game["current_turn"] = player_symbol  
          
                await self.db.update_game(game["game_id"], {  
                    "board": game["board"],  
                    "current_turn": game["current_turn"],  
                    "state": game["state"],  
                    "move_count": game["move_count"]  
                })  
          
                try:  
                    await message.edit_reply_markup(  
                        reply_markup=self.ui.get_game_board(  
                            game["board"], game["theme"], game["state"],   
                            game["current_turn"], game["players"]  
                        )  
                    )  
                except MessageNotModified:  
                    pass
        except Exception as e:
            logger.error(f"AI move execution error: {e}", exc_info=True)
            # Ensure player stats are updated even on AI error
            player = game["players"]["X"]
            if player and player["user_id"] != self.client.me.id:
                await self.db.update_user_stats(player["user_id"], "loss", self.client)
        finally:
            self.ai_tasks.pop(game["game_id"], None)
  
    async def _handle_join(self, client: Client, query: CallbackQuery):  
        user = query.from_user  
        chat = query.message.chat  
      
        game = await self.db.get_active_game(chat.id)  
        if not game or game["state"] != "waiting":  
            await query.answer("❌ Game is no longer available!", show_alert=True)  
            return  
      
        if game["players"]["X"]["user_id"] == user.id:  
            await query.answer("❌ You are already Player X!", show_alert=True)  
            return  
      
        if game["players"]["O"]:  
            await query.answer("❌ Game is full!", show_alert=True)  
            return  
      
        game["players"]["O"] = {  
            "user_id": user.id,  
            "username": user.username,  
            "first_name": user.first_name or "Warrior"  
        }  
        game["state"] = "playing"  
      
        await self.db.update_game(game["game_id"], {  
            "players": game["players"],  
            "state": "playing"  
        })  
      
        await query.message.edit_reply_markup(  
            reply_markup=self.ui.get_game_board(  
                game["board"], game["theme"],   
                game["state"], game["current_turn"],   
                game["players"]  
            )  
        )  
      
        await query.answer(f"✅ Welcome to the arena, Player O!", show_alert=True)  
      
        player_x = game["players"]["X"]["first_name"]  
        player_o = game["players"]["O"]["first_name"]  
      
        await client.send_message(  
            chat_id=query.message.chat.id,  
            text=(  
                f"⚔️ **THE BATTLE BEGINS!** ⚔️\n\n"  
                f"❌ **{player_x}** vs **{player_o}** ⭕\n\n"  
                f"🎲 **{player_x}** makes the first move!\n"  
                f"🏆 May the best strategist win!"  
            )  
        )  
  
    async def _handle_rematch(self, client: Client, query: CallbackQuery):  
        user = query.from_user  
        chat = query.message.chat  
      
        game = await self.db.get_active_game(chat.id)  
        if game and game["state"] == "finished":  
            was_player = any(  
                player and player["user_id"] == user.id   
                for player in game["players"].values()  
            )  
          
            if was_player or self.is_owner(user.id):  
                mock_message = query.message  
                mock_message.from_user = user  
                await self.handle_newgame(client, mock_message)  
                return  
      
        await query.answer("❌ Only previous players can request rematch!", show_alert=True)  
  
    async def _handle_theme_menu(self, client: Client, query: CallbackQuery):  
        user = query.from_user  
        chat = query.message.chat  
        game = await self.db.get_active_game(chat.id)  
      
        if not game:  
            await query.answer("❌ No active game!", show_alert=True)  
            return  

        # Restrict to active players  
        is_player = any(  
            player and player["user_id"] == user.id   
            for player in game["players"].values()  
        )  
        if not is_player and not self.is_owner(user.id):  
            await query.answer("⚠️ Only players in this game can change the theme!", show_alert=True)  
            return  
      
        await query.message.edit_reply_markup(  
            reply_markup=self.ui.get_theme_menu(game["theme"])  
        )  
        await query.answer()  
  
    async def _handle_set_theme(self, client: Client, query: CallbackQuery):  
        user = query.from_user  
        chat = query.message.chat  
      
        game = await self.db.get_active_game(chat.id)  
        if not game:  
            await query.answer("❌ No active game!", show_alert=True)  
            return  
          
        # Restrict to active players  
        is_player = any(  
            player and player["user_id"] == user.id   
            for player in game["players"].values()  
        )  
        if not is_player and not self.is_owner(user.id):  
            await query.answer("⚠️ Only players in this game can change the theme!", show_alert=True)  
            return  
      
        theme = query.data.replace("set_theme_", "")  
        valid_themes = ["classic", "colorful", "nature", "space", "food",   
                       "royal", "warrior", "magical", "pirate", "ninja"]  
        if theme not in valid_themes:  
            await query.answer("❌ Invalid theme!", show_alert=True)  
            return  
      
        # Update theme in memory only - no DB write for performance
        game["theme"] = theme  
      
        await query.message.edit_reply_markup(  
            reply_markup=self.ui.get_game_board(  
                game["board"], game["theme"], game["state"],   
                game["current_turn"], game["players"]  
            )  
        )  
        await query.answer(f"✅ Theme changed to {theme.title()}!")  
  
    async def _handle_back_to_game(self, client: Client, query: CallbackQuery):  
        chat = query.message.chat  
        game = await self.db.get_active_game(chat.id)  
      
        if not game:  
            await query.answer("❌ No active game!", show_alert=True)  
            return  
      
        await query.message.edit_reply_markup(  
            reply_markup=self.ui.get_game_board(  
                game["board"], game["theme"], game["state"],   
                game["current_turn"], game["players"]  
            )  
        )  
        await query.answer()  
  
    async def _handle_newgame_callback(self, client: Client, query: CallbackQuery):  
        mock_message = query.message  
        mock_message.from_user = query.from_user  
        await self.handle_newgame(client, mock_message)  
        await query.answer()  
  
    async def _handle_help_callback(self, client: Client, query: CallbackQuery):  
        help_text = (  
            "**🎮 Tic-Tac-Toe Pro - Quick Help**\n\n"  
            "**How to Play:**\n"  
            "• Tap board cells to place your mark\n"  
            "• Get 3 in a row to win!\n"  
            "• Block your opponent's moves\n\n"  
            "**Commands:**\n"  
            "/newgame - Start game\n"  
            "/ai - Challenge AI\n"  
            "/profile - Your stats\n"  
            "/leaderboard - Top players\n\n"  
            "**Tips:**\n"  
            "• Change themes during gameplay\n"  
            "• Invite friends to join"  
        )  
        await query.message.reply(help_text)  
        await query.answer()  
  
    async def _handle_toggle_enabled(self, client: Client, query: CallbackQuery):  
        user = query.from_user  
        chat = query.message.chat  
      
        if not await self.is_group_admin(chat, user.id) and not self.is_owner(user.id):  
            await query.answer("❌ Admins only!", show_alert=True)  
            return  
      
        group_settings = await self.db.get_group_settings(chat.id)  
        new_state = not group_settings["settings"]["enabled"]  
      
        await self.db.update_group_setting(chat.id, "enabled", new_state)  
        group_settings["settings"]["enabled"] = new_state  
        await query.message.edit_reply_markup(  
            reply_markup=self.ui.get_admin_menu(group_settings["settings"])  
        )  
        state = "enabled" if new_state else "disabled"  
        await query.answer(f"Bot {state}!")  
  
    async def _handle_toggle_ai(self, client: Client, query: CallbackQuery):  
        user = query.from_user  
        chat = query.message.chat  
      
        if not await self.is_group_admin(chat, user.id) and not self.is_owner(user.id):  
            await query.answer("❌ Admins only!", show_alert=True)  
            return  
      
        group_settings = await self.db.get_group_settings(chat.id)  
        new_state = not group_settings["settings"]["allow_ai"]  
      
        await self.db.update_group_setting(chat.id, "allow_ai", new_state)  
        group_settings["settings"]["allow_ai"] = new_state  
        await query.message.edit_reply_markup(  
            reply_markup=self.ui.get_admin_menu(group_settings["settings"])  
        )  
        state = "allowed" if new_state else "disabled"  
        await query.answer(f"AI {state}!")  
  
    async def _handle_toggle_pin(self, client: Client, query: CallbackQuery):  
        user = query.from_user  
        chat = query.message.chat  
      
        if not await self.is_group_admin(chat, user.id) and not self.is_owner(user.id):  
            await query.answer("❌ Admins only!", show_alert=True)  
            return  
      
        group_settings = await self.db.get_group_settings(chat.id)  
        new_state = not group_settings["settings"]["auto_pin"]  
      
        await self.db.update_group_setting(chat.id, "auto_pin", new_state)  
        group_settings["settings"]["auto_pin"] = new_state  
        await query.message.edit_reply_markup(  
            reply_markup=self.ui.get_admin_menu(group_settings["settings"])  
        )  
        state = "enabled" if new_state else "disabled"  
        await query.answer(f"Auto-pin {state}!")  
  
    async def _handle_default_theme_menu(self, client: Client, query: CallbackQuery):  
        user = query.from_user  
        chat = query.message.chat  
      
        if not await self.is_group_admin(chat, user.id) and not self.is_owner(user.id):  
            await query.answer("❌ Admins only!", show_alert=True)  
            return  
      
        themes = ["classic", "colorful", "nature", "space", "food",   
                 "royal", "warrior", "magical", "pirate", "ninja"]  
      
        buttons = []  
        row = []  
      
        for idx, theme in enumerate(themes):  
            row.append(InlineKeyboardButton(  
                theme.title(),   
                callback_data=f"set_default_theme_{theme}"  
            ))  
          
            if len(row) == 2:  
                buttons.append(row)  
                row = []  
      
        if row:  
            buttons.append(row)  
      
        buttons.append([InlineKeyboardButton("🔙 Back", callback_data="back_to_settings")])  
      
        await query.message.edit_reply_markup(reply_markup=InlineKeyboardMarkup(buttons))  
        await query.answer()  
  
    async def _handle_reset_game_callback(self, client: Client, query: CallbackQuery):  
        user = query.from_user  
        chat = query.message.chat  
      
        if not await self.is_group_admin(chat, user.id) and not self.is_owner(user.id):  
            await query.answer("❌ Admins only!", show_alert=True)  
            return  
      
        mock_message = query.message  
        mock_message.from_user = user  
        mock_message.chat = chat  
      
        await self.handle_resetgame(client, mock_message)  
        await query.answer()  
  
    async def _handle_close_menu(self, client: Client, query: CallbackQuery):  
        await query.message.delete()  
        await query.answer()


# ==================== HANDLER REGISTRATION ====================

def register_handlers(client: Client, bot: BotCore):
    """Register all Telegram handlers with proper logging"""
    logger.info("=" * 50)
    logger.info("REGISTERING COMMAND HANDLERS")
    logger.info("=" * 50)
    
    # User commands
    @client.on_message(filters.command("start"))
    async def start_handler(client, message):
        await bot.handle_start(client, message)

    @client.on_message(filters.command("newgame"))
    async def newgame_handler(client, message):
        await bot.handle_newgame(client, message)

    @client.on_message(filters.command("ai"))
    async def ai_handler(client, message):
        await bot.handle_ai(client, message)

    @client.on_message(filters.command("profile"))
    async def profile_handler(client, message):
        await bot.handle_profile(client, message)

    @client.on_message(filters.command("leaderboard"))
    async def leaderboard_handler(client, message):
        await bot.handle_leaderboard(client, message)

    @client.on_message(filters.command("help"))
    async def help_handler(client, message):
        await bot.handle_help(client, message)

    # Admin commands
    @client.on_message(filters.command("resetgame"))
    async def resetgame_handler(client, message):
        await bot.handle_resetgame(client, message)

    @client.on_message(filters.command("settings"))
    async def settings_handler(client, message):
        await bot.handle_settings(client, message)

    @client.on_message(filters.command("enablebot"))
    async def enablebot_handler(client, message):
        await bot.handle_enablebot(client, message)

    @client.on_message(filters.command("disablebot"))
    async def disablebot_handler(client, message):
        await bot.handle_disablebot(client, message)

    # Owner commands
    @client.on_message(filters.command("stats"))
    async def stats_handler(client, message):
        await bot.handle_owner_stats(client, message)

    @client.on_message(filters.command("broadcast"))
    async def broadcast_handler(client, message):
        await bot.handle_broadcast(client, message)

    @client.on_message(filters.command("active_games"))
    async def active_games_handler(client, message):
        await bot.handle_active_games(client, message)

    @client.on_message(filters.command("users"))
    async def users_handler(client, message):
        await bot.handle_users(client, message)

    @client.on_message(filters.command("groups"))
    async def groups_handler(client, message):
        await bot.handle_groups(client, message)

    @client.on_message(filters.command("restart"))
    async def restart_handler(client, message):
        await bot.handle_restart(client, message)

    @client.on_message(filters.command("shutdown"))
    async def shutdown_handler(client, message):
        await bot.handle_shutdown(client, message)

    @client.on_message(filters.command("maintenance"))
    async def maintenance_handler(client, message):
        await bot.handle_maintenance(client, message)

    @client.on_message(filters.command("block"))
    async def block_handler(client, message):
        await bot.handle_block(client, message)

    @client.on_message(filters.command("unblock"))
    async def unblock_handler(client, message):
        await bot.handle_unblock(client, message)

    # Callback queries
    @client.on_callback_query()
    async def callback_handler(client, query):
        await bot.handle_callback(client, query)
    
    logger.info("✅ All handlers registered successfully!")
    logger.info("=" * 50)


# ==================== MAIN EXECUTION ====================

def main():
    """Main entry point with proper startup sequence"""
    logger.info("=" * 60)
    logger.info("🚀 TIC-TAC-TOE BOT - STARTUP SEQUENCE INITIATED")
    logger.info("=" * 60)
    
    try:
        # Initialize components
        logger.info("1. Initializing bot components...")
        client = Client(
            "tictactoe_bot",
            api_id=settings.API_ID,
            api_hash=settings.API_HASH,
            bot_token=settings.BOT_TOKEN,
            parse_mode=enums.ParseMode.MARKDOWN
        )
        
        db = Database(settings.MONGO_URI, settings.DB_NAME)
        bot = BotCore(client, db)
        
        # Register handlers BEFORE starting client
        logger.info("2. Registering handlers...")
        register_handlers(client, bot)
        
        # Define startup sequence
        async def startup_sequence():
            """Run all startup tasks"""
            logger.info("3. Running startup sequence...")
            
            # Initialize database
            await db._create_indexes()
            logger.info("   ✅ Database indexes ready")
            
            await db._init_global_stats()
            logger.info("   ✅ Global stats initialized")
            
            # Start background tasks
            await bot.start_cleanup_task()
            logger.info("   ✅ Cleanup task started")
            
            # Notify owner
            try:
                await client.send_message(
                    chat_id=settings.OWNER_ID,
                    text="🎮 **Bot is online and ready!**"
                )
                logger.info("   ✅ Owner notification sent")
            except Exception as e:
                logger.warning(f"   ⚠️ Could not notify owner: {e}")
            
            logger.info("4. ✅ Startup complete! Bot is ready for commands.")
        
        # Main bot runner
        async def run_bot():
            """Start and run the bot"""
            logger.info("5. Starting Pyrogram client...")
            await client.start()
            logger.info("   ✅ Client started successfully")
            logger.info(f"   🤖 Bot username: @{client.me.username}")
            logger.info(f"   👑 Owner ID: {settings.OWNER_ID}")
            
            # Run startup
            await startup_sequence()
            
            # Keep bot running
            logger.info("6. Bot is now running! Press Ctrl+C to stop.")
            logger.info("=" * 60)
            await idle()
        
        # Run the bot
        client.run(run_bot())
        
    except KeyboardInterrupt:
        logger.info("\n⛔ Bot stopped by user")
        logger.info("Cleaning up tasks...")
        # Cancel any running AI tasks
        for task in getattr(bot, 'ai_tasks', {}).values():
            task.cancel()
        sys.exit(0)
        
    except Exception as e:
        logger.critical(f"❌ Fatal startup error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
