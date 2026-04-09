{{ fullname }}
{{ "=" * fullname|length }}

.. automodule:: {{ fullname }}

{% block classes %}
{% if classes %}
Classes
-------

.. autosummary::
   :nosignatures:
{% for klass in classes %}
   {{ klass }}
{% endfor %}

{% for klass in classes %}
.. autoclass:: {{ klass }}
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __call__
{% endfor %}
{% endif %}
{% endblock %}

{% block functions %}
{% if functions %}
Functions
---------

.. autosummary::
   :nosignatures:
{% for func in functions %}
   {{ func }}
{% endfor %}

{% for func in functions %}
.. autofunction:: {{ func }}
{% endfor %}
{% endif %}
{% endblock %}

{% block attributes %}
{% if attributes %}
Attributes
----------

{% for attr in attributes %}
.. autoattribute:: {{ attr }}
{% endfor %}
{% endif %}
{% endblock %}
