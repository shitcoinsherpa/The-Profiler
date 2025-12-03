"""
Database module for storing analysis history and subject profiles.
Uses SQLite for persistent storage of profiling results.
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Database file location
DB_PATH = Path(__file__).parent / "profiler_data.db"


@dataclass
class Subject:
    """Represents a profiled subject/entity."""
    id: Optional[int] = None
    name: str = ""
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""
    profile_count: int = 0


@dataclass
class ProfileRecord:
    """Represents a single profiling analysis record."""
    id: Optional[int] = None
    subject_id: Optional[int] = None
    subject_name: str = ""
    case_id: str = ""
    report_number: int = 1
    timestamp: str = ""
    video_source: str = ""
    video_metadata: str = ""  # JSON string
    models_used: str = ""  # JSON string
    processing_time: float = 0.0
    analyses: str = ""  # JSON string containing all analyses
    status: str = "completed"
    notes: str = ""


class ProfileDatabase:
    """
    Database manager for storing and retrieving profiling data.
    """

    def __init__(self, db_path: str = None):
        """
        Initialize database connection.

        Args:
            db_path: Optional custom database path
        """
        self.db_path = Path(db_path) if db_path else DB_PATH
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self):
        """Initialize database tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Create subjects table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS subjects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                notes TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')

        # Create profiles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id INTEGER,
                case_id TEXT NOT NULL UNIQUE,
                report_number INTEGER NOT NULL DEFAULT 1,
                timestamp TEXT NOT NULL,
                video_source TEXT DEFAULT '',
                video_metadata TEXT DEFAULT '{}',
                models_used TEXT DEFAULT '{}',
                processing_time REAL DEFAULT 0.0,
                essence_analysis TEXT DEFAULT '',
                multimodal_analysis TEXT DEFAULT '',
                audio_analysis TEXT DEFAULT '',
                liwc_analysis TEXT DEFAULT '',
                fbi_synthesis TEXT DEFAULT '',
                full_result TEXT DEFAULT '{}',
                status TEXT DEFAULT 'completed',
                notes TEXT DEFAULT '',
                FOREIGN KEY (subject_id) REFERENCES subjects(id) ON DELETE SET NULL
            )
        ''')

        # Create index for faster lookups
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_profiles_subject_id
            ON profiles(subject_id)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_profiles_timestamp
            ON profiles(timestamp)
        ''')

        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")

    # ==================== Subject Management ====================

    def create_subject(self, name: str, notes: str = "") -> Subject:
        """
        Create a new subject/entity.

        Args:
            name: Subject name
            notes: Optional notes about the subject

        Returns:
            Created Subject object
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        now = datetime.now().isoformat()

        try:
            cursor.execute('''
                INSERT INTO subjects (name, notes, created_at, updated_at)
                VALUES (?, ?, ?, ?)
            ''', (name.strip(), notes, now, now))

            conn.commit()
            subject_id = cursor.lastrowid

            logger.info(f"Created subject: {name} (ID: {subject_id})")

            return Subject(
                id=subject_id,
                name=name.strip(),
                notes=notes,
                created_at=now,
                updated_at=now,
                profile_count=0
            )
        except sqlite3.IntegrityError:
            # Subject already exists, return existing
            conn.close()
            return self.get_subject_by_name(name)
        finally:
            conn.close()

    def get_subject(self, subject_id: int) -> Optional[Subject]:
        """Get a subject by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT s.*, COUNT(p.id) as profile_count
            FROM subjects s
            LEFT JOIN profiles p ON s.id = p.subject_id
            WHERE s.id = ?
            GROUP BY s.id
        ''', (subject_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return Subject(
                id=row['id'],
                name=row['name'],
                notes=row['notes'],
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                profile_count=row['profile_count']
            )
        return None

    def get_subject_by_name(self, name: str) -> Optional[Subject]:
        """Get a subject by name."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT s.*, COUNT(p.id) as profile_count
            FROM subjects s
            LEFT JOIN profiles p ON s.id = p.subject_id
            WHERE LOWER(s.name) = LOWER(?)
            GROUP BY s.id
        ''', (name.strip(),))

        row = cursor.fetchone()
        conn.close()

        if row:
            return Subject(
                id=row['id'],
                name=row['name'],
                notes=row['notes'],
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                profile_count=row['profile_count']
            )
        return None

    def get_or_create_subject(self, name: str, notes: str = "") -> Subject:
        """Get existing subject or create new one."""
        subject = self.get_subject_by_name(name)
        if subject:
            return subject
        return self.create_subject(name, notes)

    def list_subjects(self, search: str = None) -> List[Subject]:
        """
        List all subjects, optionally filtered by search term.

        Args:
            search: Optional search term to filter by name

        Returns:
            List of Subject objects
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        if search:
            cursor.execute('''
                SELECT s.*, COUNT(p.id) as profile_count
                FROM subjects s
                LEFT JOIN profiles p ON s.id = p.subject_id
                WHERE s.name LIKE ?
                GROUP BY s.id
                ORDER BY s.updated_at DESC
            ''', (f'%{search}%',))
        else:
            cursor.execute('''
                SELECT s.*, COUNT(p.id) as profile_count
                FROM subjects s
                LEFT JOIN profiles p ON s.id = p.subject_id
                GROUP BY s.id
                ORDER BY s.updated_at DESC
            ''')

        rows = cursor.fetchall()
        conn.close()

        return [
            Subject(
                id=row['id'],
                name=row['name'],
                notes=row['notes'],
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                profile_count=row['profile_count']
            )
            for row in rows
        ]

    def update_subject(self, subject_id: int, name: str = None, notes: str = None) -> bool:
        """Update subject details."""
        conn = self._get_connection()
        cursor = conn.cursor()

        updates = []
        params = []

        if name is not None:
            updates.append("name = ?")
            params.append(name.strip())
        if notes is not None:
            updates.append("notes = ?")
            params.append(notes)

        if not updates:
            return False

        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(subject_id)

        cursor.execute(f'''
            UPDATE subjects SET {", ".join(updates)}
            WHERE id = ?
        ''', params)

        conn.commit()
        success = cursor.rowcount > 0
        conn.close()

        return success

    def delete_subject(self, subject_id: int, delete_profiles: bool = False) -> bool:
        """
        Delete a subject.

        Args:
            subject_id: Subject ID to delete
            delete_profiles: If True, delete associated profiles.
                           If False, profiles are orphaned.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        if delete_profiles:
            cursor.execute('DELETE FROM profiles WHERE subject_id = ?', (subject_id,))

        cursor.execute('DELETE FROM subjects WHERE id = ?', (subject_id,))

        conn.commit()
        success = cursor.rowcount > 0
        conn.close()

        return success

    # ==================== Profile Management ====================

    def save_profile(
        self,
        result: Dict,
        subject_name: str = None,
        video_source: str = "",
        notes: str = ""
    ) -> ProfileRecord:
        """
        Save a profiling result to the database.

        Args:
            result: Complete profiling result dictionary
            subject_name: Optional subject name to associate with
            video_source: Source of the video (file path or URL)
            notes: Optional notes about this profile

        Returns:
            Created ProfileRecord object
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Get or create subject if name provided
        subject_id = None
        report_number = 1

        if subject_name and subject_name.strip():
            subject = self.get_or_create_subject(subject_name.strip())
            subject_id = subject.id

            # Get next report number for this subject
            cursor.execute('''
                SELECT COALESCE(MAX(report_number), 0) + 1 as next_num
                FROM profiles WHERE subject_id = ?
            ''', (subject_id,))
            report_number = cursor.fetchone()['next_num']

            # Update subject's updated_at
            cursor.execute('''
                UPDATE subjects SET updated_at = ? WHERE id = ?
            ''', (datetime.now().isoformat(), subject_id))

        # Extract data from result
        case_id = result.get('case_id', f"PROF-{datetime.now().strftime('%Y%m%d%H%M%S')}")
        timestamp = result.get('timestamp', datetime.now().isoformat())
        processing_time = result.get('processing_time_seconds', 0.0)

        analyses = result.get('analyses', {})
        video_metadata = result.get('video_metadata', {})
        models_used = result.get('models_used', {})

        try:
            cursor.execute('''
                INSERT INTO profiles (
                    subject_id, case_id, report_number, timestamp, video_source,
                    video_metadata, models_used, processing_time,
                    essence_analysis, multimodal_analysis, audio_analysis,
                    liwc_analysis, fbi_synthesis, full_result, status, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                subject_id,
                case_id,
                report_number,
                timestamp,
                video_source,
                json.dumps(video_metadata),
                json.dumps(models_used),
                processing_time,
                analyses.get('sam_christensen_essence', ''),
                analyses.get('multimodal_behavioral', ''),
                analyses.get('audio_voice_analysis', ''),
                analyses.get('liwc_linguistic_analysis', ''),
                analyses.get('fbi_behavioral_synthesis', ''),
                json.dumps(result),
                result.get('status', 'completed'),
                notes
            ))

            conn.commit()
            profile_id = cursor.lastrowid

            logger.info(
                f"Saved profile: {case_id} "
                f"(Subject: {subject_name or 'None'}, Report #{report_number})"
            )

            return ProfileRecord(
                id=profile_id,
                subject_id=subject_id,
                subject_name=subject_name or "",
                case_id=case_id,
                report_number=report_number,
                timestamp=timestamp,
                video_source=video_source,
                processing_time=processing_time,
                status=result.get('status', 'completed'),
                notes=notes
            )

        except sqlite3.IntegrityError as e:
            logger.error(f"Failed to save profile: {e}")
            raise
        finally:
            conn.close()

    def get_profile(self, profile_id: int = None, case_id: str = None) -> Optional[Dict]:
        """
        Get a profile by ID or case_id.

        Returns:
            Full profile data as dictionary
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        if profile_id:
            cursor.execute('''
                SELECT p.*, s.name as subject_name
                FROM profiles p
                LEFT JOIN subjects s ON p.subject_id = s.id
                WHERE p.id = ?
            ''', (profile_id,))
        elif case_id:
            cursor.execute('''
                SELECT p.*, s.name as subject_name
                FROM profiles p
                LEFT JOIN subjects s ON p.subject_id = s.id
                WHERE p.case_id = ?
            ''', (case_id,))
        else:
            return None

        row = cursor.fetchone()
        conn.close()

        if row:
            return self._row_to_profile_dict(row)
        return None

    def get_profiles_for_subject(self, subject_id: int) -> List[Dict]:
        """Get all profiles for a specific subject."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT p.*, s.name as subject_name
            FROM profiles p
            LEFT JOIN subjects s ON p.subject_id = s.id
            WHERE p.subject_id = ?
            ORDER BY p.report_number DESC
        ''', (subject_id,))

        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_profile_dict(row) for row in rows]

    def list_profiles(
        self,
        subject_id: int = None,
        limit: int = 50,
        offset: int = 0,
        search: str = None
    ) -> List[Dict]:
        """
        List profiles with optional filtering.

        Args:
            subject_id: Filter by subject
            limit: Maximum number of results
            offset: Offset for pagination
            search: Search term for case_id or subject name
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        query = '''
            SELECT p.id, p.subject_id, p.case_id, p.report_number,
                   p.timestamp, p.processing_time, p.status, p.video_source,
                   s.name as subject_name
            FROM profiles p
            LEFT JOIN subjects s ON p.subject_id = s.id
            WHERE 1=1
        '''
        params = []

        if subject_id:
            query += ' AND p.subject_id = ?'
            params.append(subject_id)

        if search:
            query += ' AND (p.case_id LIKE ? OR s.name LIKE ?)'
            params.extend([f'%{search}%', f'%{search}%'])

        query += ' ORDER BY p.timestamp DESC LIMIT ? OFFSET ?'
        params.extend([limit, offset])

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                'id': row['id'],
                'subject_id': row['subject_id'],
                'subject_name': row['subject_name'] or 'Unknown',
                'case_id': row['case_id'],
                'report_number': row['report_number'],
                'timestamp': row['timestamp'],
                'processing_time': row['processing_time'],
                'status': row['status'],
                'video_source': row['video_source']
            }
            for row in rows
        ]

    def delete_profile(self, profile_id: int) -> bool:
        """Delete a profile by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('DELETE FROM profiles WHERE id = ?', (profile_id,))

        conn.commit()
        success = cursor.rowcount > 0
        conn.close()

        return success

    def _row_to_profile_dict(self, row) -> Dict:
        """Convert a database row to a full profile dictionary."""
        full_result = json.loads(row['full_result']) if row['full_result'] else {}

        return {
            'id': row['id'],
            'subject_id': row['subject_id'],
            'subject_name': row['subject_name'] or 'Unknown',
            'case_id': row['case_id'],
            'report_number': row['report_number'],
            'timestamp': row['timestamp'],
            'video_source': row['video_source'],
            'video_metadata': json.loads(row['video_metadata']) if row['video_metadata'] else {},
            'models_used': json.loads(row['models_used']) if row['models_used'] else {},
            'processing_time': row['processing_time'],
            'status': row['status'],
            'notes': row['notes'],
            'analyses': {
                'sam_christensen_essence': row['essence_analysis'],
                'multimodal_behavioral': row['multimodal_analysis'],
                'audio_voice_analysis': row['audio_analysis'],
                'liwc_linguistic_analysis': row['liwc_analysis'],
                'fbi_behavioral_synthesis': row['fbi_synthesis'],
            },
            'full_result': full_result
        }

    # ==================== Statistics ====================

    def get_stats(self) -> Dict:
        """Get database statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) as count FROM subjects')
        subject_count = cursor.fetchone()['count']

        cursor.execute('SELECT COUNT(*) as count FROM profiles')
        profile_count = cursor.fetchone()['count']

        cursor.execute('''
            SELECT AVG(processing_time) as avg_time
            FROM profiles WHERE status = 'completed'
        ''')
        avg_time = cursor.fetchone()['avg_time'] or 0

        cursor.execute('''
            SELECT s.name, COUNT(p.id) as count
            FROM subjects s
            JOIN profiles p ON s.id = p.subject_id
            GROUP BY s.id
            ORDER BY count DESC
            LIMIT 5
        ''')
        top_subjects = [
            {'name': row['name'], 'count': row['count']}
            for row in cursor.fetchall()
        ]

        conn.close()

        return {
            'total_subjects': subject_count,
            'total_profiles': profile_count,
            'average_processing_time': round(avg_time, 2),
            'top_subjects': top_subjects
        }


# Global database instance
_db_instance = None


def get_database() -> ProfileDatabase:
    """Get the global database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = ProfileDatabase()
    return _db_instance
