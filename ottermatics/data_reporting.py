"""Mapping a polymorphic-valued vertical table as a dictionary.

Builds upon the dictlike.py example to also add differently typed
columns to the "fact" table, e.g.::

  Table('properties', metadata
        Column('owner_id', Integer, ForeignKey('owner.id'),
               primary_key=True),
        Column('key', UnicodeText),
        Column('type', Unicode(16)),
        Column('int_value', Integer),
        Column('char_value', UnicodeText),
        Column('bool_value', Boolean),
        Column('decimal_value', Numeric(10,2)))

For any given properties row, the value of the 'type' column will point to the
'_value' column active for that row.

This example approach uses exactly the same dict mapping approach as the
'dictlike' example.  It only differs in the mapping for vertical rows.  Here,
we'll use a @hybrid_property to build a smart '.value' attribute that wraps up
reading and writing those various '_value' columns and keeps the '.type' up to
date.

"""

from sqlalchemy import event
from sqlalchemy import literal_column
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm.interfaces import PropComparator

from ottermatics.data import *

class ProxiedDictMixin(object):
    """Adds obj[key] access to a mapped class.

    This class basically proxies dictionary access to an attribute
    called ``_proxied``.  The class which inherits this class
    should have an attribute called ``_proxied`` which points to a dictionary.

    """

    def __len__(self):
        return len(self._proxied)

    def __iter__(self):
        return iter(self._proxied)

    def __getitem__(self, key):
        return self._proxied[key]

    def __contains__(self, key):
        return key in self._proxied

    def __setitem__(self, key, value):
        self._proxied[key] = value

    def __delitem__(self, key):
        del self._proxied[key]

class PolymorphicVerticalProperty(object):
    """A key/value pair with polymorphic value storage.

    The class which is mapped should indicate typing information
    within the "info" dictionary of mapped Column objects; see
    the NumericEntry mapping below for an example.

    """

    def __init__(self, key, value=None):
        self.key = key
        self.value = value

    @hybrid_property
    def value(self):
        fieldname, discriminator = self.type_map[self.type]
        if fieldname is None:
            return None
        else:
            return getattr(self, fieldname)

    @value.setter
    def value(self, value):
        py_type = type(value)
        fieldname, discriminator = self.type_map[py_type]

        self.type = discriminator
        if fieldname is not None:
            setattr(self, fieldname, value)

    @value.deleter
    def value(self):
        self._set_value(None)

    @value.comparator
    class value(PropComparator):
        """A comparator for .value, builds a polymorphic comparison via
        CASE."""

        def __init__(self, cls):
            self.cls = cls

        def _case(self):
            pairs = set(self.cls.type_map.values())
            whens = [
                (
                    literal_column("'%s'" % discriminator),
                    cast(getattr(self.cls, attribute), String),
                )
                for attribute, discriminator in pairs
                if attribute is not None
            ]
            return case(whens, self.cls.type, null())

        def __eq__(self, other):
            return self._case() == cast(other, String)

        def __ne__(self, other):
            return self._case() != cast(other, String)

    def __repr__(self):
        return "<%s %r=%r>" % (self.__class__.__name__, self.key, self.value)


@event.listens_for(
    PolymorphicVerticalProperty, "mapper_configured", propagate=True
)
def on_new_class(mapper, cls_):
    """Look for Column objects with type info in them, and work up
    a lookup table."""

    info_dict = {}
    info_dict[type(None)] = (None, "none")
    info_dict["none"] = (None, "none")

    for k in mapper.c.keys():
        col = mapper.c[k]
        if "type" in col.info:
            python_type, discriminator = col.info["type"]
            info_dict[python_type] = (k, discriminator)
            info_dict[discriminator] = (k, discriminator)
    cls_.type_map = info_dict


if __name__ == "__main__":
    from sqlalchemy import (
        Column,
        Integer,
        Unicode,
        ForeignKey,
        UnicodeText,
        and_,
        or_,
        String,
        Boolean,
        cast,
        null,
        case,
        create_engine,
    )
    from sqlalchemy.orm import relationship, Session
    from sqlalchemy.orm.collections import attribute_mapped_collection
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.ext.associationproxy import association_proxy

    Base = declarative_base()

    #Polymorphic or just numeric?
    class NumericEntry(PolymorphicVerticalProperty, Base):
        """A numeric value with key name"""

        __tablename__ = "numeric_entries"

        tabulation_id = Column(ForeignKey("tabulation.id"), primary_key=True)
        key = Column(Unicode(64), primary_key=True)
        value = Column(Numeric)
        #type = Column(Unicode(16))

        # add information about storage for different types
        # in the info dictionary of Columns
        # int_value = Column(Integer, info={"type": (int, "integer")})
        # char_value = Column(UnicodeText, info={"type": (str, "string")})
        # boolean_value = Column(Boolean, info={"type": (bool, "boolean")})

    class TabulationResult(ProxiedDictMixin, Base):
        """an TabulationResult"""

        __tablename__ = "tabulation"

        id = Column(Integer, primary_key=True)
        name = Column(Unicode(100))

        facts = relationship(
            "NumericEntry", collection_class=attribute_mapped_collection("key")
        )

        _proxied = association_proxy(
            "facts",
            "value",
            creator=lambda key, value: NumericEntry(key=key, value=value),
        )

        def __init__(self, name):
            self.name = name

        def __repr__(self):
            return "Tabulation(%r)" % self.name

        @classmethod
        def with_characteristic(self, key, value):
            return self.facts.any(key=key, value=value)

    #engine = create_engine("sqlite://", echo=True)
    db = DBConnection(database_name='report_test',host='localhost',user='postgres',passd='***REMOVED***')
    engine = db.engine

    Base.metadata.create_all(engine)
    session = Session(engine)

    stoat = TabulationResult("stoat")
    stoat["color"] = 19.1
    stoat["cuteness"] = 7
    stoat["weasel-like"] = 43

    session.add(stoat)
    session.commit()

    # critter = session.query(TabulationResult).filter(TabulationResult.name == "stoat").one()
    # print(critter["color"])
    # print(critter["cuteness"])

    # print("changing cuteness value and type:")
    # critter["cuteness"] = "very cute"

    # session.commit()

    marten = TabulationResult("marten")
    marten["cuteness"] = 5123
    marten["weasel-like"] = 123
    marten["poisonous"] = 124
    session.add(marten)

    shrew = TabulationResult("shrew")
    shrew["cuteness"] = 5
    shrew["weasel-like"] = 32
    shrew["poisonous"] = 12

    session.add(shrew)
    session.commit()

    # q = session.query(TabulationResult).filter(
    #     TabulationResult.facts.any(
    #         and_(NumericEntry.key == "weasel-like", NumericEntry.value == True)
    #     )
    # )
    # print("weasel-like animals", q.all())

    # q = session.query(TabulationResult).filter(
    #     TabulationResult.with_characteristic("weasel-like", True)
    # )
    # print("weasel-like animals again", q.all())

    # q = session.query(TabulationResult).filter(
    #     TabulationResult.with_characteristic("poisonous", False)
    # )
    # print("animals with poisonous=False", q.all())

    # q = session.query(TabulationResult).filter(
    #     or_(
    #         TabulationResult.with_characteristic("poisonous", False),
    #         ~TabulationResult.facts.any(NumericEntry.key == "poisonous"),
    #     )
    # )
    # print("non-poisonous animals", q.all())

    # q = session.query(TabulationResult).filter(TabulationResult.facts.any(NumericEntry.value == 5))
    # print("any animal with a .value of 5", q.all())